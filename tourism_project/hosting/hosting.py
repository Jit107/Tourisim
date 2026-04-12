from huggingface_hub import HfApi, create_repo
import os
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "jit0107/tourism-customer-predictor-space" # New Hugging Face Space repo ID
repo_type = "space"

# Step 1: Check if the space exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/deployment",     # Correct local folder containing deployment files
    repo_id=repo_id,                              # Target Hugging Face Space
    repo_type=repo_type,                          # Type is 'space'
    path_in_repo="",                              # Upload to the root of the space
)
