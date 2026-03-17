"""Storage helpers for local paths and S3 URIs.

The experiment tooling uses these helpers to keep data access consistent across
training, validation, and orchestration scripts.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

try:
    import fsspec
except Exception as exc:  # pragma: no cover - handled at runtime
    fsspec = None
    _fsspec_import_error = exc
else:
    _fsspec_import_error = None

try:
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
except Exception as exc:  # pragma: no cover - handled at runtime
    ds = None
    pafs = None
    _pyarrow_import_error = exc
else:
    _pyarrow_import_error = None


PathLike = Union[str, os.PathLike[str]]

DEFAULT_CACHE_ROOT = Path(
    os.environ.get("CLIMB_S3_CACHE_DIR", str(Path.home() / ".cache" / "climb_s3"))
)


def is_s3_uri(path: PathLike) -> bool:
    return str(path).startswith("s3://")


def split_s3_uri(uri: PathLike) -> Tuple[str, str]:
    parsed = urlparse(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Not an S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def normalize_uri(path: PathLike) -> str:
    return str(path)


def default_cache_dir(cache_dir: Optional[PathLike] = None) -> Path:
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _require_pyarrow_fs():
    if pafs is None or ds is None:
        raise ImportError(
            "pyarrow with dataset/fs support is required for parquet/S3 access. "
            f"Import error: {_pyarrow_import_error}"
        )


def _require_fsspec():
    if fsspec is None:
        raise ImportError(
            "fsspec is required for remote file materialization. "
            f"Import error: {_fsspec_import_error}"
        )


def _s3_fsspec():
    _require_fsspec()
    try:
        return fsspec.filesystem("s3")
    except Exception as exc:  # pragma: no cover - depends on optional s3fs install
        raise ImportError(
            "Install s3fs to access S3 through fsspec."
        ) from exc


def s3_filesystem():
    _require_pyarrow_fs()
    return pafs.S3FileSystem(region=os.environ.get("AWS_REGION"))


def _s3_selector_path(uri: str) -> str:
    bucket, key = split_s3_uri(uri)
    return f"{bucket}/{key}".rstrip("/")


def _s3_uri_from_info_path(info_path: str) -> str:
    return f"s3://{info_path.lstrip('/')}"


def list_data_files(
    paths: Union[PathLike, Sequence[PathLike]],
    suffixes: Sequence[str] = (".pkl", ".parquet"),
) -> List[str]:
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    resolved: List[str] = []

    for raw in items:
        path = normalize_uri(raw)
        if is_s3_uri(path):
            if any(path.endswith(sfx) for sfx in suffixes):
                resolved.append(path)
                continue

            selector_path = _s3_selector_path(path)
            fs = s3_filesystem()
            infos = fs.get_file_info(pafs.FileSelector(selector_path, recursive=True))
            for info in infos:
                if info.type != pafs.FileType.File:
                    continue
                if suffixes and not info.path.endswith(tuple(suffixes)):
                    continue
                resolved.append(_s3_uri_from_info_path(info.path))
            continue

        p = Path(path)
        if p.is_dir():
            for suffix in suffixes:
                resolved.extend(sorted(str(x) for x in p.glob(f"*{suffix}")))
            continue
        if p.exists():
            resolved.append(str(p))

    return sorted(dict.fromkeys(resolved))


def path_exists(path: PathLike) -> bool:
    target = normalize_uri(path)
    if is_s3_uri(target):
        try:
            if any(target.endswith(sfx) for sfx in (".pkl", ".parquet", ".json", ".yaml", ".yml")):
                return bool(_s3_fsspec().exists(target))
            return bool(list_data_files(target))
        except Exception:
            return False
    return Path(target).exists()


def cache_key_path(uri: str, cache_dir: Optional[PathLike] = None) -> Path:
    bucket, key = split_s3_uri(uri)
    cache_root = default_cache_dir(cache_dir)
    return cache_root / bucket / key


def materialize_path(path: PathLike, cache_dir: Optional[PathLike] = None) -> str:
    target = normalize_uri(path)
    if not is_s3_uri(target):
        return target

    local_path = cache_key_path(target, cache_dir=cache_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    _require_fsspec()
    fd, tmp_name = tempfile.mkstemp(prefix="climb_", suffix=".tmp", dir=str(local_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with fsspec.open(target, "rb") as src, tmp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        os.replace(tmp_path, local_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return str(local_path)


def materialize_paths(
    paths: Iterable[PathLike],
    cache_dir: Optional[PathLike] = None,
) -> List[str]:
    return [materialize_path(path, cache_dir=cache_dir) for path in paths]


def materialize_tokenizer_dir(path: PathLike, cache_dir: Optional[PathLike] = None) -> str:
    target = normalize_uri(path)
    if not is_s3_uri(target):
        return target

    local_root = cache_key_path(target.rstrip("/"), cache_dir=cache_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    ):
        remote = f"{target.rstrip('/')}/{filename}"
        try:
            if path_exists(remote):
                materialize_path(remote, cache_dir=cache_dir)
        except Exception:
            continue
    return str(local_root)


def parquet_dataset(path: PathLike):
    _require_pyarrow_fs()
    target = normalize_uri(path)
    if is_s3_uri(target):
        fs = s3_filesystem()
        return ds.dataset(_s3_selector_path(target), format="parquet", filesystem=fs)
    return ds.dataset(target, format="parquet")
