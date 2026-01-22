from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

api = HfApi()

api.create_repo(
    repo_id="fozga/szpadel",
    repo_type="dataset",
    exist_ok=True,
)

api.upload_folder(
    folder_path="./results",
    repo_id="fozga/szpadel",
)

api.upload_folder(
    folder_path="./experiment_logs",
    repo_id="fozga/szpadel",
)
