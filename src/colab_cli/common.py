# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer

from colab_cli.auth import AuthProvider, get_credentials
from colab_cli.client import Client, Prod, RuntimeProxyInfo
from colab_cli.history import HistoryLogger
from colab_cli.state import SessionState, StateStore, SettingsStore

# Headroom so a token doesn't expire mid-command.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
ASSIGNMENT_RECHECK_DELAY_SECONDS = 2.0
STALE_SESSION_MAX_MISSES = 3


def _apply_proxy_info(s: SessionState, info: RuntimeProxyInfo):
    s.token = info.token
    s.url = info.url
    s.token_expires_at = info.expires_at()


class State:
    def __init__(self):
        self.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
        self.config_path = None
        self.logtostderr = False
        self.auth_provider = AuthProvider.OAUTH2
        self._client = None
        self._store = None
        self._stale_store = None
        self._settings_store = None
        self._history = None
        self._sessions = None

    @property
    def store(self):
        if self._store is None:
            self._store = StateStore(self.config_path)
        return self._store

    @property
    def stale_store(self):
        if self._stale_store is None:
            if self.config_path:
                root, ext = os.path.splitext(self.config_path)
                stale_path = (
                    f"{root}.stale{ext}" if ext else f"{self.config_path}.stale"
                )
            else:
                stale_path = os.path.expanduser(
                    "~/.config/colab-cli/stale-sessions.json"
                )
            self._stale_store = StateStore(stale_path)
        return self._stale_store

    @property
    def settings_store(self):
        if self._settings_store is None:
            # We don't currently allow overriding settings path via CLI,
            # but we could if needed. For now, use default.
            self._settings_store = SettingsStore()
        return self._settings_store

    @property
    def history(self):
        if self._history is None:
            self._history = HistoryLogger()
        return self._history

    @property
    def client(self):
        if self._client is None:
            creds = get_credentials(
                self.client_oauth_config, provider=self.auth_provider
            )
            self._client = Client(Prod(), creds)
        return self._client

    def _list_assignments_with_retry(self, required_endpoints: set[str]):
        """List assignments, confirming local-endpoint misses with one retry."""
        first = self.client.list_assignments()
        first_by_endpoint = {a.endpoint: a for a in first}
        missing = required_endpoints - set(first_by_endpoint)
        if not missing:
            return first, first_by_endpoint, set()

        time.sleep(ASSIGNMENT_RECHECK_DELAY_SECONDS)
        second = self.client.list_assignments()
        second_by_endpoint = {a.endpoint: a for a in second}
        confirmed_missing = missing - set(second_by_endpoint)

        # Use the latest listing, but retain local endpoints that were present in
        # the first listing so a contradictory second listing cannot destroy
        # their only local handle in this invocation.
        stable_by_endpoint = dict(second_by_endpoint)
        for endpoint in required_endpoints - missing:
            if endpoint not in stable_by_endpoint:
                stable_by_endpoint[endpoint] = first_by_endpoint[endpoint]

        assignments = list(second)
        seen = {a.endpoint for a in assignments}
        for endpoint, assignment in stable_by_endpoint.items():
            if endpoint not in seen:
                assignments.append(assignment)
        return assignments, stable_by_endpoint, confirmed_missing

    def _mark_session_stale(self, s: SessionState) -> bool:
        """Move/update a missing assignment in the stale store.

        Returns True when the record reached the consecutive-miss threshold and
        was finally removed.
        """
        existing = self.stale_store.get(s.name)
        if existing is not None:
            s = existing
        s.assignment_misses += 1
        if s.stale_since is None:
            s.stale_since = datetime.now(timezone.utc)

        self.store.remove(s.name)
        if self._sessions and s.name in self._sessions:
            del self._sessions[s.name]

        if s.assignment_misses >= STALE_SESSION_MAX_MISSES:
            self.stale_store.remove(s.name)
            self.history.log_event(
                s.name,
                "session_terminated",
                {"reason": "confirmed_missing", "endpoint": s.endpoint},
            )
            typer.echo(
                f"[colab] Session '{s.name}' missing from server assignments "
                f"for {STALE_SESSION_MAX_MISSES} consecutive checks; removed "
                f"stale local record (endpoint {s.endpoint})."
            )
            return True

        self.stale_store.add(s)
        self.history.log_event(
            s.name,
            "session_stale",
            {"endpoint": s.endpoint, "misses": s.assignment_misses},
        )
        typer.echo(
            f"[colab] Session '{s.name}' not in server assignments; kept as "
            f"stale (endpoint {s.endpoint}, miss {s.assignment_misses}/"
            f"{STALE_SESSION_MAX_MISSES}). Run `colab sessions` to reconcile."
        )
        return False

    def _recover_stale_session(self, s: SessionState, assignment) -> SessionState:
        _apply_proxy_info(s, assignment.runtime_proxy_info)
        s.assignment_misses = 0
        s.stale_since = None
        self.store.add(s)
        self.stale_store.remove(s.name)
        if self._sessions is not None:
            self._sessions[s.name] = s
        self.history.log_event(
            s.name,
            "session_recovered",
            {"endpoint": s.endpoint, "by": "assignment_reconciliation"},
        )
        typer.echo(
            f"[colab] Recovered stale session '{s.name}' from endpoint {s.endpoint}."
        )
        return s

    def prune_session(self, name: str):
        """Removes a session from local active and stale state."""
        self.store.remove(name)
        if self.stale_store.get(name) is not None:
            self.stale_store.remove(name)
        if self._sessions and name in self._sessions:
            del self._sessions[name]
        self.history.log_event(name, "session_terminated", {"reason": "pruned"})

    def get_session(
        self, name: str, ignore_missing_session: bool = False
    ) -> Optional[SessionState]:
        """Load a session and conservatively reconcile its server assignment."""
        s = self.store.get(name)
        stale = None if s is not None else self.stale_store.get(name)
        candidate = s or stale

        needs_reconcile = bool(
            candidate
            and (
                stale is not None
                or not candidate.token_expires_at
                or candidate.token_expires_at - datetime.now(timezone.utc)
                <= TOKEN_REFRESH_MARGIN
            )
        )
        if candidate and needs_reconcile:
            _, by_endpoint, confirmed_missing = self._list_assignments_with_retry(
                {candidate.endpoint}
            )
            assignment = by_endpoint.get(candidate.endpoint)
            if assignment is not None:
                if stale is not None:
                    s = self._recover_stale_session(candidate, assignment)
                else:
                    _apply_proxy_info(candidate, assignment.runtime_proxy_info)
                    candidate.assignment_misses = 0
                    candidate.stale_since = None
                    self.store.add(candidate)
                    s = candidate
            elif candidate.endpoint in confirmed_missing:
                self._mark_session_stale(candidate)
                s = None

        if s is None and not ignore_missing_session:
            if self.stale_store.get(name) is not None:
                typer.echo(
                    f"[colab] Session '{name}' is stale: its endpoint is not "
                    "currently in server assignments. Run `colab sessions` to reconcile."
                )
            else:
                typer.echo(f"[colab] Session '{name}' not found.")
            raise typer.Exit(1)
        return s

    def sync_sessions(self):
        stale_sessions = self.stale_store.list()
        if self._sessions is not None and not stale_sessions:
            return self._sessions, self.client.list_assignments()

        local_sessions = self.store.list()
        if not local_sessions and not stale_sessions:
            self._sessions = {}
            try:
                assignments = self.client.list_assignments()
            except SystemExit:
                assignments = []
            return self._sessions, assignments

        required_endpoints = {
            s.endpoint for s in [*local_sessions.values(), *stale_sessions.values()]
        }
        assignments, by_endpoint, confirmed_missing = self._list_assignments_with_retry(
            required_endpoints
        )

        self._sessions = dict(local_sessions)
        for name, s in list(local_sessions.items()):
            assignment = by_endpoint.get(s.endpoint)
            if assignment is not None:
                _apply_proxy_info(s, assignment.runtime_proxy_info)
                s.assignment_misses = 0
                s.stale_since = None
                self.store.add(s)
            elif s.endpoint in confirmed_missing:
                self._mark_session_stale(s)

        for name, s in list(stale_sessions.items()):
            assignment = by_endpoint.get(s.endpoint)
            if assignment is not None:
                self._recover_stale_session(s, assignment)
            elif s.endpoint in confirmed_missing:
                self._mark_session_stale(s)

        return self._sessions, assignments

    def resolve_session(self, session_name: Optional[str]) -> str:
        if session_name:
            return session_name

        # Check local store first to avoid hitting the backend (and triggering auth) if we don't have to
        local_sessions = self.store.list()
        if not local_sessions:
            stale_sessions = self.stale_store.list()
            if len(stale_sessions) == 1:
                name = next(iter(stale_sessions))
                typer.echo(
                    f"[colab] Using unique stale session '{name}' for reconciliation."
                )
                return name
            if len(stale_sessions) > 1:
                typer.echo(
                    "[colab] Error: Multiple stale sessions found. Specify one with -s: "
                    + ", ".join(stale_sessions)
                )
                raise typer.Exit(1)
            typer.echo(
                "[colab] Error: No active sessions found. Create one with 'colab new'."
            )
            raise typer.Exit(1)

        # If we have local sessions, we need to sync to make sure they are still valid.
        # This will trigger auth if valid credentials are not present.
        sessions, _ = self.sync_sessions()
        active_names = list(sessions.keys())

        if len(active_names) == 1:
            name = active_names[0]
            typer.echo(f"[colab] Using unique session '{name}'.")
            return name
        elif len(active_names) > 1:
            typer.echo(
                f"[colab] Error: Multiple active sessions found. Specify one with -s: {', '.join(active_names)}"
            )
            raise typer.Exit(1)
        else:
            typer.echo(
                "[colab] Error: No active sessions found. Create one with 'colab new'."
            )
            raise typer.Exit(1)


state = State()


def setup_logging(log_to_stderr: bool):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    requests_log = logging.getLogger("urllib3")
    # urllib3's DEBUG lines include full request URLs. Contents API requests
    # carry the short-lived runtime proxy token in the query string, so never
    # persist urllib3 wire logging in the CLI's default debug log.
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = True

    log_dir = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(log_dir, mode=0o700, exist_ok=True)
    if os.name != "nt":
        os.chmod(log_dir, 0o700)
    log_path = os.path.join(log_dir, "colab.log")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    os.close(fd)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    if log_to_stderr:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(stream_handler)
