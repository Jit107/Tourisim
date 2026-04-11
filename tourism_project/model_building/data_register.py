from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

#Set your token inside the script for standalone execution
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") # Assuming HF_TOKEN is already set as an environment variable or will be in CI/CD
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "jit0107/Tourism"
repo_type = "dataset"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

print("Dataset registration complete. tourism.csv should already be uploaded to the HF dataset repo.")
