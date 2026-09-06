"""Common constants and utility helpers for the project."""

from __future__ import annotations

from pathlib import Path

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B"
DATASET_SOURCE_PATH = "./data/data.csv"
DATASET_PATH = "./data/dataset_clean.jsonl"
MICROSERVICES_MODEL_DIR = "./microservices_model_qlora"

IGNORED_DIRECTORIES = {
    "node_modules",
    ".git",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    "test",
    "tests",
    "__tests__",
}

VALID_FILE_EXTENSIONS = {".ts"}


def normalize_text(text: str) -> str:
    """Normalize line endings and trim the content.

    Args:
        text (str): Raw source content.

    Returns:
        str: Cleaned text ready to be saved in the dataset.
    """
    return text.replace("\u2028", "\n").replace("\u2029", "\n").strip()


def is_valid_source_file(file_name: str) -> bool:
    """Return whether the file matches the allowed TypeScript dataset rules.

    Args:
        file_name (str): File name to validate.

    Returns:
        bool: True when the file is valid for processing.
    """
    return any(file_name.endswith(extension) for extension in VALID_FILE_EXTENSIONS) or file_name == "Dockerfile"


def ensure_output_directory(file_path: str) -> None:
    """Create the parent directory for a generated output file.

    Args:
        file_path (str): Destination path of the file to write.

    Returns:
        None: This function does not return a value.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
