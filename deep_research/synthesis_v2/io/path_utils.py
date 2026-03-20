from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def find_repo_root(start: Union[str, Path]) -> Path:
    """
    从 start 向上查找包含 .git 的目录作为 repo root。
    若找不到，则退化为 start 的最上层父目录（尽量稳定，不依赖 cwd）。
    """
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent

    for parent in (p, *p.parents):
        if (parent / ".git").exists():
            return parent

    # fallback：尽量回到工程目录（不使用 cwd）
    return p.parents[-1] if p.parents else p


def resolve_under_repo(path_like: Union[str, Path], *, anchor_file: Optional[Union[str, Path]] = None) -> Path:
    """
    如果是相对路径，则把它解析到 repo root 下；绝对路径原样返回。
    """
    raw = Path(path_like)
    if raw.is_absolute():
        return raw

    anchor = Path(anchor_file).resolve() if anchor_file is not None else Path(__file__).resolve()
    repo_root = find_repo_root(anchor)
    return (repo_root / raw).resolve()

