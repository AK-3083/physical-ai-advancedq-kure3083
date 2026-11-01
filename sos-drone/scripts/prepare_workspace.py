"""コンテナ起動時に生成物とGazebo描画キャッシュの所有者をubuntuへ揃える。"""

from itertools import chain
import os
from pathlib import Path
import pwd


def _prepare_directory(directory):
    if os.geteuid() != 0:
        raise PermissionError("Workspace preparation must run as root during container startup")
    user = pwd.getpwnam("ubuntu")
    if directory.is_symlink():
        raise ValueError(f"Output directory must not be a symbolic link: {directory}")
    directory.mkdir(exist_ok=True)
    for path in chain((directory,), directory.rglob("*")):
        if path.lstat().st_uid == 0:
            os.chown(path, user.pw_uid, user.pw_gid, follow_symlinks=False)


def prepare_workspace(workspace=Path("/workspace/sos")):
    """root所有の生成物だけを修正し、コードや外部リンク先は変更しない。"""
    for name in ("runs", "assets", "models", "deliverables"):
        _prepare_directory(workspace / name)


def prepare_gazebo_cache(directory=Path("/tmp/gazebo-ubuntu-rtshaderlibcache")):
    """Gazeboの描画キャッシュをubuntuユーザーが読み書きできるようにする。"""
    _prepare_directory(directory)


if __name__ == "__main__":
    prepare_workspace()
    prepare_gazebo_cache()
    print("Workspace outputs and Gazebo shader cache are ready for the ubuntu user.")
