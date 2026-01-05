from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="Qwen3-1.7B",
    endpoint="https://hf-mirror.com", # 使用镜像
    resume_download=True,
    local_dir_use_symlinks=False
)