import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_dataset(repo_id, repo_type="dataset", local_dir=".", max_retries=5, retry_delay=5):
    """
    Downloads a dataset or model from the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        repo_type (str): The type of repository. Defaults to "dataset".
        local_dir (str): Optional local directory to store the files. If None, uses the default cache directory.
        max_retries (int): Maximum number of retries in case of failure. Defaults to 5.
        retry_delay (int): Delay (in seconds) between retries. Defaults to 5 seconds.

    Returns:
        str: Path to the downloaded repository.
    """
    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "force_download": True,
        "max_workers": 1,
    }
    if local_dir is not None:
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        download_kwargs["local_dir"] = str(local_path)

    retries = 0
    while retries < max_retries:
        try:
            print(f"Starting download from repo: {repo_id} (type: {repo_type})")
            download_path = snapshot_download(**download_kwargs)
            print(f"Download complete. Files are located at: {download_path}")
            return download_path
        except Exception as e:
            retries += 1
            print(f"Error occurred while downloading {repo_id}: {e}")
            if retries < max_retries:
                print(f"Retrying download ({retries}/{max_retries}) after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Exceeded maximum retries for {repo_id}. Skipping...")
                raise


if __name__ == "__main__":
    print("Starting download script...")

    repo_ids = [
        "ESA-PhiLab-Edge/LPL-Burned-Area-Seg",
    ]

    local_directory = Path("data") / "hf"

    for repo_id in repo_ids:
        download_dataset(repo_id=repo_id, repo_type="dataset", local_dir=local_directory)
