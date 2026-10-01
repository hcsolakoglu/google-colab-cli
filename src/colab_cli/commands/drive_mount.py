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

from typing import Optional

import typer
from typing_extensions import Annotated

from colab_cli.drive_mount import (
    DriveMountAuthError,
    drive_mount_status,
    login_drive_mount,
    logout_drive_mount,
    mount_drive_persistent,
)

app = typer.Typer(
    help="Persistent DriveFS auth and mounts.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback()
def drive_mount(
    ctx: typer.Context,
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    path: Annotated[
        str, typer.Option("--path", help="Remote Drive mount path")
    ] = "/content/drive",
):
    """Mount Drive without repeated browser consent after one-time login."""
    if ctx.invoked_subcommand is not None:
        return
    from colab_cli.common import state

    name = state.resolve_session(session)
    try:
        mount_drive_persistent(name, path)
    except DriveMountAuthError as exc:
        typer.echo(f"[colab] {exc}", err=True)
        raise typer.Exit(1)


@app.command("login")
def login():
    """Authorize DriveFS once and persist a refresh token locally."""
    try:
        auth = login_drive_mount()
    except DriveMountAuthError as exc:
        typer.echo(f"[colab] {exc}", err=True)
        raise typer.Exit(1)
    account = f" for {auth['email']}" if auth.get("email") else ""
    typer.echo(f"[colab] Persistent Drive authorization saved{account}.")


@app.command("status")
def status():
    """Show persistent DriveFS configuration and authorization state."""
    try:
        info = drive_mount_status()
    except DriveMountAuthError as exc:
        typer.echo(f"[colab] {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Configured: {'yes' if info['configured'] else 'no'}")
    typer.echo(f"Authorized: {'yes' if info['authorized'] else 'no'}")
    if info["authorized"] and info["configured"]:
        typer.echo(
            f"OAuth client matches token: {'yes' if info['client_matches'] else 'no'}"
        )
    if info.get("email"):
        typer.echo(f"Account: {info['email']}")
    typer.echo(f"Credential file: {info['credential_file']}")


@app.command("logout")
def logout():
    """Revoke the stored Drive refresh token and remove local credentials."""
    try:
        revoked = logout_drive_mount()
    except DriveMountAuthError as exc:
        typer.echo(f"[colab] {exc}", err=True)
        raise typer.Exit(1)
    if revoked:
        typer.echo("[colab] Drive credential revoked and removed.")
    else:
        typer.echo(
            "[colab] Local Drive credential removed, but Google revocation could "
            "not be confirmed. Revoke the app grant in your Google account if needed.",
            err=True,
        )


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="drive-mount")
