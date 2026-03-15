import os
import sys
from pathlib import Path
from typing import Optional


def _is_uva_root(path: Path) -> bool:
    return path.is_dir() and (path / "unified_video_action").is_dir()


def resolve_uva_root(anchor_file: Optional[str] = None) -> Optional[str]:
    """
    Resolve UVA repo root that contains the `unified_video_action` package.

    Priority:
      1) Explicit env vars
      2) Nearby candidate directories relative to anchor_file
    """
    env_candidates = [
        os.environ.get("JIT_UVA_ROOT"),
        os.environ.get("UVA_ROOT"),
        os.environ.get("UVA_BO_ROOT"),
    ]
    for item in env_candidates:
        if not item:
            continue
        p = Path(item).expanduser().resolve()
        if _is_uva_root(p):
            return str(p)

    if anchor_file is None:
        anchor = Path.cwd().resolve()
    else:
        anchor = Path(anchor_file).resolve().parent

    nearby = [
        anchor / ".." / "unified_video_action",
        anchor / ".." / "uva-bo",
        anchor / ".." / ".." / "unified_video_action",
        anchor / ".." / ".." / "uva-bo",
        anchor / "..",
        anchor / ".." / "..",
    ]
    for cand in nearby:
        p = cand.resolve()
        if _is_uva_root(p):
            return str(p)

    return None


def ensure_uva_on_sys_path(
    anchor_file: Optional[str] = None,
    prepend: bool = False,
) -> Optional[str]:
    root = resolve_uva_root(anchor_file=anchor_file)
    if root:
        if root in sys.path:
            # keep only one entry; optionally move to front
            sys.path = [p for p in sys.path if p != root]
        if prepend:
            sys.path.insert(0, root)
        else:
            # append to avoid overriding local modules unexpectedly
            sys.path.append(root)
    return root
