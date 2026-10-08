"""Environment and path helpers for iterative fine-tuning."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

_UNRESOLVED_ENV_VAR_PATTERN = re.compile(r"\$(\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def expand_value(value: str) -> str:
    """Expand environment variables and user-home shortcuts in a string."""
    return os.path.expanduser(os.path.expandvars(value))


def ensure_no_unresolved_env_vars(value: str) -> None:
    """Fail fast when a configured path still contains unresolved env vars."""
    if _UNRESOLVED_ENV_VAR_PATTERN.search(value):
        raise ValueError(
            "Encountered an unresolved environment variable while resolving a path: "
            f"{value!r}. Make sure the variable is defined in your shell or .env file."
        )


def resolve_local_path(
    pathlike: str | Path,
    *,
    base_dir: Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a possibly relative local path against a base directory."""
    expanded = expand_value(str(pathlike))
    ensure_no_unresolved_env_vars(expanded)
    path = Path(expanded)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_model_reference(reference: str | Path, *, base_dir: Path | None = None) -> str:
    """Resolve a local model/checkpoint path while leaving Hub-style refs untouched."""
    if isinstance(reference, Path):
        return str(resolve_local_path(reference, base_dir=base_dir))

    expanded = expand_value(reference)
    ensure_no_unresolved_env_vars(expanded)
    looks_local = expanded.startswith(("/", "./", "../", "~")) or expanded != reference
    if looks_local:
        return str(resolve_local_path(expanded, base_dir=base_dir))
    return expanded


def maybe_load_env_file(env_file: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    """Load a .env file if configured and present."""
    if env_file is None:
        return None

    env_path = resolve_local_path(env_file, base_dir=base_dir)
    if not env_path.exists():
        return None
    
    load_dotenv(env_path, override=False)
    return env_path
