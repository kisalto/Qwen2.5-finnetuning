# Python
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Third party
import pandas as pd
import requests
from tqdm import tqdm

# Local
from utils import (
    DATASET_PATH,
    DATASET_SOURCE_PATH,
    IGNORED_DIRECTORIES,
    MICROSERVICES_MODEL_DIR,
    VALID_FILE_EXTENSIONS,
    ensure_output_directory,
    is_valid_source_file,
    normalize_text,
)

MAX_THREADS = 12
REPOSITORY_COLUMN_NAME = "URL"
DATASET_LOCK = threading.Lock()


def fetch_repository_archive(repository_url: str) -> bytes | None:
    """Download the repository archive for the provided GitHub URL.

    Args:
        repository_url (str): Repository base URL.

    Returns:
        bytes | None: ZIP archive bytes or None if the download fails.
    """
    for branch_name in ("main", "master"):
        archive_url = f"{repository_url}/zipball/{branch_name}"
        try:
            response = requests.get(archive_url, timeout=15)
            if response.status_code == 200:
                return response.content
        except requests.RequestException:
            continue
    return None


def save_clean_dataset_entry(text: str) -> None:
    """Persist a valid text chunk into the dataset file.

    Args:
        text (str): Clean string to be saved.

    Returns:
        None: This function does not return a value.
    """
    if len(text.strip()) < 50:
        return

    dataset_payload = {"text": text.strip()}
    with DATASET_LOCK:
        with open(DATASET_PATH, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(dataset_payload, ensure_ascii=False) + "\n")


def process_repository(repository_url: str) -> None:
    """Download, filter and store valid TypeScript files from a repository.

    Args:
        repository_url (str): Repository URL to process.

    Returns:
        None: This function does not return a value.
    """
    archive_data = fetch_repository_archive(repository_url)
    if archive_data is None:
        return

    try:
        with zipfile.ZipFile(io.BytesIO(archive_data)) as zip_file:
            for archive_path in zip_file.namelist():
                if archive_path.endswith("/"):
                    continue

                path_parts = archive_path.split("/")
                if any(directory_name in path_parts for directory_name in IGNORED_DIRECTORIES):
                    continue

                file_name = path_parts[-1]
                if not is_valid_source_file(file_name):
                    continue

                try:
                    with zip_file.open(archive_path) as file_handle:
                        raw_text = file_handle.read().decode("utf-8", errors="ignore")
                except (UnicodeDecodeError, OSError):
                    continue

                normalized_text = normalize_text(raw_text)
                if normalized_text:
                    save_clean_dataset_entry(normalized_text)
    except zipfile.BadZipFile:
        return


def load_repository_links(csv_path: str) -> list[str]:
    """Load the list of repositories eligible for processing.

    Args:
        csv_path (str): CSV file path with repository data.

    Returns:
        list[str]: Filtered list of repository URLs.
    """
    dataset_frame = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
    filtered_frame = dataset_frame[
        (dataset_frame["Is a Microservices?"] == "Yes")
        & dataset_frame["languages"].astype(str).str.contains("typescript", case=False, na=False)
    ]

    return (
        filtered_frame[REPOSITORY_COLUMN_NAME]
        .dropna()
        .astype(str)
        .str.strip()
        .str.rstrip("/")
        .tolist()
    )


def main() -> None:
    """Run the dataset-creation pipeline.

    Returns:
        None: This function does not return a value.
    """
    ensure_output_directory(DATASET_PATH)
    repository_links = load_repository_links(DATASET_SOURCE_PATH)

    if not repository_links:
        print("No valid repositories were found to process.")
        return

    print(f"Found {len(repository_links)} valid repositories.")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_repository, link): link for link in repository_links}
        for _ in tqdm(as_completed(futures), total=len(repository_links), desc="Progress", unit="repo"):
            pass

    print(f"Dataset completed successfully in '{DATASET_PATH}'.")


if __name__ == "__main__":
    main()