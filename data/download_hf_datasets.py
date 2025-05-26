# pip install huggingface_hub
# This script downloads datasets or models from the Hugging Face Hub using the `huggingface_hub` library.
import os
import time
from huggingface_hub import snapshot_download

# Set the environment variable
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def download_dataset(repo_id, repo_type="dataset", local_dir="./", max_retries=5, retry_delay=5):
    """
    Downloads a dataset or model from the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        repo_type (str): The type of repository. Defaults to "dataset".
        local_dir (str): Optional local directory to store the files. If None, uses the default cache directory.
        max_retries (int): Maximum number of retries in case of failure. Defaults to 3.
        retry_delay (int): Delay (in seconds) between retries. Defaults to 5 seconds.

    Returns:
        str: Path to the downloaded repository.
    """
    retries = 0
    while retries < max_retries:
        try:
            print(f"Starting download from repo: {repo_id} (type: {repo_type})")
            download_path = snapshot_download(repo_id=repo_id, 
                                              repo_type=repo_type, 
                                              force_download=True,
                                              max_workers=1,
                                              force_download=True,
                                              local_dir=local_dir)
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
    
    # Define repository details
    # List of datasets to download
    repo_ids = [
        "distilbert-base-uncased",  # Example pre-trained model
        "huggingface/datasets-examples",  # Example dataset repository
        # Add more repositories as needed
    ]
    
    # Create a directory for downloaded files
    local_directory = "."
    
    # Download repositories one at a time
    for repo_id in repo_ids:
        try:
            download_dataset(repo_id=repo_id, repo_type="dataset", local_dir=local_directory)
        except Exception as e:
            print(f"Skipping {repo_id} due to an error: {e}")