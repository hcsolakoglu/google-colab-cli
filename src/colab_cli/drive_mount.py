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
"""Persistent Google DriveFS authentication and mounting.

This implementation is native to the Python CLI. It relies on Google Colab's
Apache-2.0 google.colab.drive._mount(..., ephemeral=False) contract and a
minimal runtime-local GCE metadata-compatible token endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from google.auth.transport.requests import AuthorizedSession
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
import typer

from colab_cli.runtime import ColabRuntime

DRIVE_MOUNT_AUTH_FILE = Path(
    os.path.expanduser("~/.config/colab-cli/drive-mount-auth.json")
)
DRIVE_MOUNT_CLIENT_FILE = Path(
    os.path.expanduser("~/.config/colab-cli/drive-mount-client.json")
)
DRIVEFS_CLIENT_ID_ENV = "COLAB_DRIVEFS_CLIENT_ID"
DRIVEFS_CLIENT_SECRET_ENV = "COLAB_DRIVEFS_CLIENT_SECRET"
DRIVE_MOUNT_SCOPES = (
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive",
)
DRIVE_MOUNT_TIMEOUT_SEC = 600
_GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


class DriveMountAuthError(RuntimeError):
    """Persistent DriveFS authentication is missing or unusable."""


def _normalize_scopes(scopes: Optional[Iterable[str] | str]) -> set[str]:
    if not scopes:
        return set()
    if isinstance(scopes, str):
        return set(scopes.split())
    return set(scopes)


def _read_desktop_client_file(path: Path) -> tuple[str, str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DriveMountAuthError(
            f"OAuth Desktop client file not found: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DriveMountAuthError(
            f"Could not read OAuth Desktop client file {path}: {exc}"
        ) from exc

    installed = payload.get("installed") if isinstance(payload, dict) else None
    if not isinstance(installed, dict):
        raise DriveMountAuthError(
            f"{path} is not a Google OAuth Desktop client JSON (missing installed config)."
        )
    client_id = str(installed.get("client_id") or "").strip()
    client_secret = str(installed.get("client_secret") or "").strip()
    if not client_id or not client_secret:
        raise DriveMountAuthError(
            f"{path} is missing OAuth Desktop client_id/client_secret."
        )
    return client_id, client_secret, payload


def _drive_client_credentials(
    client_config_path: Optional[Path] = None,
) -> tuple[str, str]:
    client_id = os.environ.get(DRIVEFS_CLIENT_ID_ENV, "").strip()
    client_secret = os.environ.get(DRIVEFS_CLIENT_SECRET_ENV, "").strip()
    if client_id and client_secret:
        return client_id, client_secret
    if client_id or client_secret:
        raise DriveMountAuthError(
            f"Set both {DRIVEFS_CLIENT_ID_ENV} and {DRIVEFS_CLIENT_SECRET_ENV}, or neither."
        )

    path = client_config_path or DRIVE_MOUNT_CLIENT_FILE
    file_id, file_secret, _ = _read_desktop_client_file(path)
    return file_id, file_secret


def persistent_drive_configured() -> bool:
    client_id = os.environ.get(DRIVEFS_CLIENT_ID_ENV, "").strip()
    client_secret = os.environ.get(DRIVEFS_CLIENT_SECRET_ENV, "").strip()
    if client_id or client_secret:
        return bool(client_id and client_secret)
    try:
        _read_desktop_client_file(DRIVE_MOUNT_CLIENT_FILE)
        return True
    except DriveMountAuthError:
        return False


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(encoded)


def load_drive_mount_auth() -> Optional[dict[str, Any]]:
    try:
        data = json.loads(DRIVE_MOUNT_AUTH_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise DriveMountAuthError(
            f"Could not read {DRIVE_MOUNT_AUTH_FILE}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not data.get("refresh_token"):
        raise DriveMountAuthError(
            f"{DRIVE_MOUNT_AUTH_FILE} does not contain a refresh token."
        )
    return data


def persistent_drive_authorized() -> bool:
    try:
        auth = load_drive_mount_auth()
        client_id, _ = _drive_client_credentials()
    except DriveMountAuthError:
        return False
    return bool(auth and auth.get("client_id") == client_id)


def _desktop_client_config(client_id: str, client_secret: str) -> dict[str, Any]:
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def login_drive_mount(
    client_config_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Authorize once and persist the refresh token plus a private Desktop client config."""
    if client_config_path is not None:
        client_id, client_secret, client_payload = _read_desktop_client_file(
            client_config_path
        )
        _write_private_json(DRIVE_MOUNT_CLIENT_FILE, client_payload)
    else:
        client_id, client_secret = _drive_client_credentials()
    flow = InstalledAppFlow.from_client_config(
        _desktop_client_config(client_id, client_secret),
        scopes=list(DRIVE_MOUNT_SCOPES),
    )
    creds = flow.run_local_server(
        host="127.0.0.1",
        port=0,
        open_browser=True,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message=(
            "Open this URL in a browser to authorize persistent Colab Drive mounting:\n{url}"
        ),
        success_message="Drive authorization complete. You may close this window.",
    )
    granted = _normalize_scopes(
        getattr(creds, "granted_scopes", None) or getattr(creds, "scopes", None)
    )
    missing = set(DRIVE_MOUNT_SCOPES) - granted
    if missing:
        raise DriveMountAuthError(
            "Drive authorization did not grant required scopes: "
            + ", ".join(sorted(missing))
        )
    if not creds.refresh_token:
        raise DriveMountAuthError(
            "Google did not return a refresh token. Revoke the existing app grant "
            "for this OAuth client and run drive-mount login again."
        )

    email = None
    try:
        response = AuthorizedSession(creds).get(_USERINFO_URL, timeout=20)
        response.raise_for_status()
        email = response.json().get("email")
    except Exception:
        email = None

    payload = {
        "client_id": client_id,
        "refresh_token": creds.refresh_token,
        "scopes": sorted(granted),
        "email": email,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_private_json(DRIVE_MOUNT_AUTH_FILE, payload)
    return payload


def logout_drive_mount() -> bool:
    """Revoke the stored refresh token when possible, then remove local credentials."""
    auth = load_drive_mount_auth()
    if not auth:
        return True

    revoked = False
    try:
        response = requests.post(
            _GOOGLE_REVOKE_URL,
            params={"token": auth["refresh_token"]},
            timeout=30,
        )
        revoked = response.ok
    except requests.RequestException:
        revoked = False

    try:
        DRIVE_MOUNT_AUTH_FILE.unlink(missing_ok=True)
    except OSError as exc:
        raise DriveMountAuthError(
            f"Could not remove {DRIVE_MOUNT_AUTH_FILE}: {exc}"
        ) from exc
    return revoked


def drive_mount_status() -> dict[str, Any]:
    auth = load_drive_mount_auth()
    configured = persistent_drive_configured()
    current_id = None
    if configured:
        try:
            current_id, _ = _drive_client_credentials()
        except DriveMountAuthError:
            configured = False
    return {
        "configured": configured,
        "authorized": bool(auth),
        "client_matches": bool(
            auth and current_id and auth.get("client_id") == current_id
        ),
        "email": auth.get("email") if auth else None,
        "credential_file": str(DRIVE_MOUNT_AUTH_FILE),
        "client_file": str(DRIVE_MOUNT_CLIENT_FILE),
    }


def _persistent_mount_code() -> str:
    # Secrets arrive through input_reply instead of source code. Combined with
    # store_history=False this keeps them out of both CLI and IPython history.
    return r"""
def _colab_cli_start_persistent_drive():
    import builtins as _builtins
    import json as _json
    import os as _os
    import threading as _threading
    import time as _time
    import urllib.parse as _urlparse
    import urllib.request as _urlrequest
    from http.server import BaseHTTPRequestHandler as _BaseHTTPRequestHandler
    from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
    from google.colab import drive as _drive

    _payload = _json.loads(input("__COLAB_CLI_DRIVEFS_AUTH__"))
    _path = _payload.pop("path")
    _client_id = _payload.pop("client_id")
    _client_secret = _payload.pop("client_secret")
    _refresh_token = _payload.pop("refresh_token")
    _email = _payload.pop("email", None) or "default"
    _scopes = tuple(_payload.pop("scopes"))
    _cache = {"token": None, "expires_at": 0.0}

    def _access_token():
        now = _time.time()
        if _cache["token"] and _cache["expires_at"] - now > 300:
            return _cache["token"], max(1, int(_cache["expires_at"] - now))
        body = _urlparse.urlencode(
            {
                "client_id": _client_id,
                "client_secret": _client_secret,
                "refresh_token": _refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        req = _urlrequest.Request(
            "https://oauth2.googleapis.com/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with _urlrequest.urlopen(req, timeout=30) as response:
            token_data = _json.loads(response.read().decode("utf-8"))
        token = token_data["access_token"]
        expires_in = int(token_data.get("expires_in", 3600))
        _cache["token"] = token
        _cache["expires_at"] = _time.time() + expires_in
        return token, expires_in

    class _MetadataHandler(_BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def _send(self, status, body, content_type="text/plain; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Metadata-Flavor", "Google")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = _urlparse.urlsplit(self.path).path.rstrip("/")
            base = "/computeMetadata/v1"
            if path == base or path == "":
                self._send(200, "instance/\n")
                return
            if path == base + "/instance":
                self._send(200, "service-accounts/\n")
                return
            if path == base + "/instance/service-accounts":
                self._send(200, "default/\n")
                return
            if path == base + "/instance/service-accounts/default":
                self._send(200, "email\nscopes\ntoken\n")
                return
            if path.endswith("/service-accounts/default/email"):
                self._send(200, _email)
                return
            if path.endswith("/service-accounts/default/scopes"):
                self._send(200, "\n".join(_scopes))
                return
            if path.endswith("/service-accounts/default/token"):
                token, expires_in = _access_token()
                self._send(
                    200,
                    _json.dumps(
                        {
                            "access_token": token,
                            "expires_in": expires_in,
                            "token_type": "Bearer",
                            "scope": " ".join(_scopes),
                        }
                    ),
                    "application/json",
                )
                return
            self._send(404, "not found")

    previous = getattr(_builtins, "_colab_cli_drivefs_metadata_server", None)
    if previous is not None:
        try:
            previous.shutdown()
            previous.server_close()
        except Exception:
            pass

    server = _ThreadingHTTPServer(("127.0.0.1", 0), _MetadataHandler)
    server.daemon_threads = True
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _builtins._colab_cli_drivefs_metadata_server = server
    _os.environ["TBE_CREDS_ADDR"] = "http://127.0.0.1:%d" % server.server_port

    if not hasattr(_drive, "_mount"):
        raise RuntimeError(
            "Installed google.colab.drive has no non-ephemeral _mount helper."
        )
    _drive._mount(_path, ephemeral=False)
    print("COLAB_CLI_PERSISTENT_DRIVE_MOUNTED:" + _path)


_colab_cli_start_persistent_drive()
del _colab_cli_start_persistent_drive
"""


def _send_colab_reply(deserialize_msg: dict[str, Any], wsclient: Any) -> None:
    msg_id = deserialize_msg.get("metadata", {}).get("colab_msg_id")
    reply = wsclient.session.msg(
        "input_reply",
        {"value": {"type": "colab_reply", "colab_msg_id": msg_id}},
    )
    if "header" in deserialize_msg:
        reply["parent_header"] = deserialize_msg["header"]
    wsclient.stdin_channel.send(reply)


def configure_persistent_drive_hook(runtime: ColabRuntime, session: Any) -> None:
    """Skip repeated ephemeral Drive consent after this session was persistently mounted."""
    if not getattr(session, "persistent_drive_mounted", False):
        return
    previous_hook = runtime.colab_request_hook

    def hook(deserialize_msg: dict[str, Any], wsclient: Any) -> bool:
        content = deserialize_msg.get("content", {})
        auth_type = content.get("request", {}).get("authType")
        if auth_type == "dfs_ephemeral":
            _send_colab_reply(deserialize_msg, wsclient)
            return True
        if previous_hook:
            return bool(previous_hook(deserialize_msg, wsclient))
        return False

    runtime.colab_request_hook = hook


def mount_drive_persistent(session_name: str, path: str = "/content/drive") -> None:
    """Mount DriveFS on one runtime using locally persisted OAuth credentials."""
    from colab_cli.common import state

    client_id, client_secret = _drive_client_credentials()
    auth = load_drive_mount_auth()
    if not auth:
        raise DriveMountAuthError(
            "Persistent DriveFS is not authorized. Run colab drive-mount login first."
        )
    if auth.get("client_id") != client_id:
        raise DriveMountAuthError(
            "Stored DriveFS refresh token belongs to a different OAuth client. "
            "Run colab drive-mount logout, then colab drive-mount login."
        )

    s = state.get_session(session_name)
    runtime = ColabRuntime(
        s.url,
        s.token,
        session_name=s.name,
        history=state.history,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
    )
    secret_payload = json.dumps(
        {
            "path": path,
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": auth["refresh_token"],
            "email": auth.get("email"),
            "scopes": auth.get("scopes") or list(DRIVE_MOUNT_SCOPES),
        }
    )

    s.persistent_drive_mounted = False
    state.store.add(s)

    try:
        state.history.log_event(
            s.name,
            "automation",
            {"op": "drive-mount", "path": path, "persistent": True},
        )
        outputs = runtime.execute_code(
            _persistent_mount_code(),
            allow_stdin=True,
            stdin_hook=lambda _prompt: secret_payload,
            timeout=DRIVE_MOUNT_TIMEOUT_SEC,
            store_history=False,
        )
        errors = [o for o in outputs if o.get("output_type") == "error"]
        if errors:
            error = errors[-1]
            raise DriveMountAuthError(
                f"{error.get('ename', 'DriveMountError')}: "
                f"{error.get('evalue', 'persistent Drive mount failed')}"
            )
        s.kernel_id = runtime.kernel_id
        s.session_id = runtime.session_id
        s.persistent_drive_mounted = True
        s.persistent_drive_path = path
        state.store.add(s)
        state.history.log_event(
            s.name,
            "automation_result",
            {"op": "drive-mount", "path": path, "persistent": True, "ok": True},
        )
        for output in outputs:
            output_text = output.get("text")
            if output_text:
                typer.echo(output_text, nl=not output_text.endswith("\n"))
    finally:
        runtime.stop()
