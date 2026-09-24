# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import os
import uuid
from unittest.mock import MagicMock

import requests

from colab_cli.auth import _write_private_text
from colab_cli.client import Assignment, Client, GetAssignmentResponse, Prod
from colab_cli.common import setup_logging
from colab_cli.history import HistoryLogger
from colab_cli.state import SessionState, StateStore


def test_client_debug_log_does_not_persist_auth_or_colab_tokens():
    session = MagicMock()
    response = MagicMock()
    response.ok = True
    response.status_code = 200
    response.reason = "OK"
    response.text = (
        ')]}\'\n{"token":"response-secret","nested":{"token":"nested-secret"}}'
    )
    response.headers = {"Set-Cookie": "session=response-cookie-secret"}
    response.request.headers = {
        "Authorization": "Bearer bearer-secret",
        "X-Goog-Colab-Token": "xsrf-secret",
        "X-Colab-Runtime-Proxy-Token": "runtime-secret",
        "Accept": "application/json",
    }
    session.request.return_value = response

    logger = MagicMock(spec=logging.Logger)
    client = Client(Prod(), session, logger=logger)
    client._issue_request("https://colab.research.google.com/tun/m/test")

    rendered = "\n".join(str(call) for call in logger.debug.call_args_list)
    for secret in (
        "bearer-secret",
        "xsrf-secret",
        "runtime-secret",
        "response-cookie-secret",
        "response-secret",
        "nested-secret",
    ):
        assert secret not in rendered


def test_assign_recovers_committed_assignment_after_read_timeout(mocker):
    client = Client(Prod(), MagicMock())
    nbh = uuid.uuid4()
    pending = GetAssignmentResponse(
        acc="NONE",
        nbh="some_nbh",
        token="xsrf",
        variant="DEFAULT",
    )
    recovered = Assignment(
        endpoint="ep-recovered",
        runtimeProxyInfo={
            "token": "proxy-token",
            "tokenExpiresInSeconds": 3600,
            "url": "https://runtime.example",
        },
    )
    get_assignment = mocker.patch.object(
        client, "_get_assignment", side_effect=[pending, pending, recovered]
    )
    mocker.patch.object(
        client,
        "_post_assignment",
        side_effect=requests.exceptions.ReadTimeout("timeout"),
    )
    mocker.patch("colab_cli.client.time.sleep")

    result = client.assign(nbh)

    assert result == recovered
    assert get_assignment.call_count == 3


def test_state_store_creates_private_state_file(tmp_path):
    path = tmp_path / "sessions.json"
    old_umask = os.umask(0o022)
    try:
        StateStore(str(path)).add(
            SessionState(name="s", token="secret", url="u", endpoint="e")
        )
    finally:
        os.umask(old_umask)

    assert path.stat().st_mode & 0o777 == 0o600


def test_history_logger_creates_private_history_file(tmp_path):
    old_umask = os.umask(0o022)
    try:
        logger = HistoryLogger(log_dir=str(tmp_path))
        logger.log_event("s", "execution", {"code": "print(1)", "outputs": []})
    finally:
        os.umask(old_umask)

    assert (tmp_path / "s.jsonl").stat().st_mode & 0o777 == 0o600


def test_oauth_token_writer_creates_private_file(tmp_path):
    path = tmp_path / "token.json"
    old_umask = os.umask(0o022)
    try:
        _write_private_text(str(path), '{"refresh_token":"secret"}')
    finally:
        os.umask(old_umask)

    assert path.stat().st_mode & 0o777 == 0o600


def test_setup_logging_does_not_enable_urllib3_wire_debug(tmp_path, mocker):
    root = logging.getLogger()
    urllib3_logger = logging.getLogger("urllib3")
    old_root_handlers = list(root.handlers)
    old_level = urllib3_logger.level
    mocker.patch("colab_cli.common.os.path.expanduser", return_value=str(tmp_path))
    try:
        setup_logging(False)
        assert urllib3_logger.level >= logging.WARNING
        assert (tmp_path / "colab.log").stat().st_mode & 0o777 == 0o600
    finally:
        for handler in list(root.handlers):
            if handler not in old_root_handlers:
                root.removeHandler(handler)
                handler.close()
        urllib3_logger.setLevel(old_level)
