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

import json
from unittest.mock import MagicMock
import webbrowser

from colab_cli.drive_mount import (
    DRIVE_MOUNT_SCOPES,
    persistent_drive_authorized,
    persistent_drive_configured,
    _run_drive_oauth_flow,
    configure_persistent_drive_hook,
    drive_mount_status,
    login_drive_mount,
    logout_drive_mount,
    mount_drive_persistent,
)
from colab_cli.state import SessionState


def _configure(monkeypatch):
    monkeypatch.setenv("COLAB_DRIVEFS_CLIENT_ID", "client-id")
    monkeypatch.setenv("COLAB_DRIVEFS_CLIENT_SECRET", "client-secret")


def _client_payload():
    return {
        "installed": {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def _auth_payload():
    return {
        "client_id": "client-id",
        "refresh_token": "refresh-secret",
        "scopes": list(DRIVE_MOUNT_SCOPES),
        "email": "user@example.com",
        "created_at": "2026-10-01T00:00:00+00:00",
    }


def test_oauth_flow_falls_back_when_no_desktop_browser(mocker):
    first_flow = MagicMock()
    first_flow.run_local_server.side_effect = webbrowser.Error("no browser")
    second_flow = MagicMock()
    creds = MagicMock()
    second_flow.run_local_server.return_value = creds
    factory = mocker.patch(
        "colab_cli.drive_mount._new_drive_oauth_flow",
        side_effect=[first_flow, second_flow],
    )

    result = _run_drive_oauth_flow("client-id", "client-secret")

    assert result is creds
    assert factory.call_count == 2
    assert first_flow.run_local_server.call_args.kwargs["open_browser"] is True
    assert second_flow.run_local_server.call_args.kwargs["open_browser"] is False


def test_login_persists_refresh_token_privately_without_client_secret(
    tmp_path, monkeypatch, mocker
):
    _configure(monkeypatch)
    auth_file = tmp_path / "drive-mount-auth.json"
    monkeypatch.setattr("colab_cli.drive_mount.DRIVE_MOUNT_AUTH_FILE", auth_file)

    creds = MagicMock()
    creds.refresh_token = "refresh-secret"
    creds.granted_scopes = set(DRIVE_MOUNT_SCOPES)
    flow = MagicMock()
    flow.run_local_server.return_value = creds
    flow_cls = mocker.patch("colab_cli.drive_mount.InstalledAppFlow")
    flow_cls.from_client_config.return_value = flow

    userinfo = MagicMock()
    userinfo.get.return_value = MagicMock(
        json=lambda: {"email": "user@example.com"},
        raise_for_status=lambda: None,
    )
    mocker.patch("colab_cli.drive_mount.AuthorizedSession", return_value=userinfo)

    result = login_drive_mount()

    saved = json.loads(auth_file.read_text())
    assert result["refresh_token"] == "refresh-secret"
    assert saved["refresh_token"] == "refresh-secret"
    assert saved["client_id"] == "client-id"
    assert "client_secret" not in saved
    assert auth_file.stat().st_mode & 0o777 == 0o600
    kwargs = flow.run_local_server.call_args.kwargs
    assert kwargs["access_type"] == "offline"
    assert kwargs["prompt"] == "consent"


def test_login_with_client_config_copies_private_config_for_future_mounts(
    tmp_path, monkeypatch, mocker
):
    monkeypatch.delenv("COLAB_DRIVEFS_CLIENT_ID", raising=False)
    monkeypatch.delenv("COLAB_DRIVEFS_CLIENT_SECRET", raising=False)
    source = tmp_path / "downloaded-client.json"
    source.write_text(json.dumps(_client_payload()))
    client_file = tmp_path / "drive-mount-client.json"
    auth_file = tmp_path / "drive-mount-auth.json"
    monkeypatch.setattr("colab_cli.drive_mount.DRIVE_MOUNT_CLIENT_FILE", client_file)
    monkeypatch.setattr("colab_cli.drive_mount.DRIVE_MOUNT_AUTH_FILE", auth_file)

    creds = MagicMock()
    creds.refresh_token = "refresh-secret"
    creds.granted_scopes = set(DRIVE_MOUNT_SCOPES)
    flow = MagicMock()
    flow.run_local_server.return_value = creds
    flow_cls = mocker.patch("colab_cli.drive_mount.InstalledAppFlow")
    flow_cls.from_client_config.return_value = flow
    userinfo = MagicMock()
    userinfo.get.return_value = MagicMock(
        json=lambda: {"email": "user@example.com"},
        raise_for_status=lambda: None,
    )
    mocker.patch("colab_cli.drive_mount.AuthorizedSession", return_value=userinfo)

    login_drive_mount(source)

    assert json.loads(client_file.read_text()) == _client_payload()
    assert client_file.stat().st_mode & 0o777 == 0o600
    assert persistent_drive_configured() is True
    assert persistent_drive_authorized() is True


def test_status_reports_client_mismatch(tmp_path, monkeypatch):
    _configure(monkeypatch)
    auth_file = tmp_path / "drive-mount-auth.json"
    auth_file.write_text(json.dumps({**_auth_payload(), "client_id": "old-client"}))
    monkeypatch.setattr("colab_cli.drive_mount.DRIVE_MOUNT_AUTH_FILE", auth_file)

    status = drive_mount_status()

    assert status["configured"] is True
    assert status["authorized"] is True
    assert status["client_matches"] is False


def test_logout_revokes_then_removes_local_token(tmp_path, monkeypatch, mocker):
    _configure(monkeypatch)
    auth_file = tmp_path / "drive-mount-auth.json"
    auth_file.write_text(json.dumps(_auth_payload()))
    monkeypatch.setattr("colab_cli.drive_mount.DRIVE_MOUNT_AUTH_FILE", auth_file)

    response = MagicMock()
    response.ok = True
    post = mocker.patch("colab_cli.drive_mount.requests.post", return_value=response)

    assert logout_drive_mount() is True
    assert not auth_file.exists()
    assert post.call_args.kwargs["params"] == {"token": "refresh-secret"}


def test_mount_passes_secrets_via_redacted_stdin_not_source_or_history(
    tmp_path, monkeypatch, mocker
):
    _configure(monkeypatch)
    auth_file = tmp_path / "drive-mount-auth.json"
    auth_file.write_text(json.dumps(_auth_payload()))
    monkeypatch.setattr("colab_cli.drive_mount.DRIVE_MOUNT_AUTH_FILE", auth_file)

    session = SessionState(
        name="s",
        token="runtime-proxy",
        url="https://runtime.example",
        endpoint="ep",
    )
    state = MagicMock()
    state.get_session.return_value = session
    mocker.patch("colab_cli.common.state", state)

    runtime = MagicMock()
    runtime.kernel_id = "kid"
    runtime.session_id = "sid"
    runtime.execute_code.return_value = [{"text": "mounted\n"}]
    runtime_cls = mocker.patch(
        "colab_cli.drive_mount.ColabRuntime", return_value=runtime
    )

    mount_drive_persistent("s", "/content/drive")

    runtime_cls.assert_called_once()
    code = runtime.execute_code.call_args.args[0]
    kwargs = runtime.execute_code.call_args.kwargs
    assert "refresh-secret" not in code
    assert "client-secret" not in code
    assert kwargs["store_history"] is False
    assert kwargs["allow_stdin"] is True
    secret_payload = json.loads(kwargs["stdin_hook"]("prompt"))
    assert secret_payload["refresh_token"] == "refresh-secret"
    assert secret_payload["client_secret"] == "client-secret"

    rendered_history = "\n".join(str(c) for c in state.history.log_event.call_args_list)
    assert "refresh-secret" not in rendered_history
    assert "client-secret" not in rendered_history
    assert session.persistent_drive_mounted is True
    assert session.persistent_drive_path == "/content/drive"
    state.store.add.assert_called_with(session)


def test_persistent_hook_acknowledges_ephemeral_drive_auth_without_browser():
    runtime = MagicMock()
    runtime.colab_request_hook = None
    session = SessionState(
        name="s",
        token="t",
        url="u",
        endpoint="e",
        persistent_drive_mounted=True,
    )
    wsclient = MagicMock()
    wsclient.session.msg.return_value = {
        "content": {"value": {"type": "colab_reply", "colab_msg_id": "m1"}},
        "header": {},
    }
    message = {
        "content": {"request": {"authType": "dfs_ephemeral"}},
        "metadata": {"colab_msg_id": "m1"},
        "header": {"msg_id": "parent"},
    }

    configure_persistent_drive_hook(runtime, session)

    assert runtime.colab_request_hook(message, wsclient) is True
    wsclient.stdin_channel.send.assert_called_once()
    reply = wsclient.stdin_channel.send.call_args.args[0]
    assert reply["content"]["value"] == {
        "type": "colab_reply",
        "colab_msg_id": "m1",
    }
