"""Upload AdversaRiskQA dataset and evaluation framework to Hugging Face Hub."""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()


def upload_dataset(
    folder_path: Path,
    repo_id: str = "dudderr/adversarisk-qa",
    token: str | None = None,
    private: bool = False,
) -> None:
    """Upload prepared dataset folder to Hugging Face Hub."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        login(token=token, add_to_git_credential=False)
        api = HfApi(token=token)
    else:
        api = HfApi()

    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"[+] Authenticated with Hugging Face as: {username}")
    except Exception as e:
        print(f"[-] Authentication failed: {e}")
        print("    Please log in via 'hf auth login' or provide a token via --token or HF_TOKEN env var.")
        sys.exit(1)

    print(f"[+] Creating / verifying dataset repository '{repo_id}'...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"[+] Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"[-] Failed to create repository '{repo_id}': {e}")
        print("    Make sure your token has 'write' permissions for the specified namespace.")
        sys.exit(1)

    print(f"[+] Uploading directory '{folder_path}' to '{repo_id}'...")
    commit_info = api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload AdversaRiskQA benchmark dataset & evaluation framework",
    )

    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n[SUCCESS] Successfully uploaded to Hugging Face Hub!")
    print(f"Repository URL: {hf_url}")
    print(f"Commit: {commit_info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument(
        "--folder",
        type=str,
        default="hf_export",
        help="Path to folder to upload (default: hf_export)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="dudderr/adversarisk-qa",
        help="Target Hugging Face repository ID (default: dudderr/adversarisk-qa)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face write token",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private instead of public",
    )
    args = parser.parse_args()

    upload_dataset(
        folder_path=Path(args.folder),
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )

