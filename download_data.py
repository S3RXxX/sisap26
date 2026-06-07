from huggingface_hub import hf_hub_download

# Download a specific dataset
file_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5",
    repo_type="dataset"
)

# Download a config file
config_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="wikipedia-small/config.json",
    repo_type="dataset"
)

print(f"Data path: {file_path}")
print(f"Config path: {config_path}")