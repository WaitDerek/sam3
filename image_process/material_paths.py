from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent
CONFIG_DIR = TOOL_DIR / "configs"
WORK_ROOT = TOOL_DIR
DATA_DIR = WORK_ROOT / "dataset"
OUT_DIR = WORK_ROOT / "out"
CHANGAN_DIR = WORK_ROOT / "changan"


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "image_process":
        return REPO_ROOT / path
    return WORK_ROOT / path


def resolve_work_path(path: str | Path) -> Path:
    return resolve_repo_path(path)


def resolve_config_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    candidates = [
        TOOL_DIR / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if path.parts and path.parts[0] == "configs":
        return TOOL_DIR / path
    if path.parts and path.parts[0] == "image_process":
        return REPO_ROOT / path
    return TOOL_DIR / path


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(WORK_ROOT))
    except ValueError:
        pass
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
