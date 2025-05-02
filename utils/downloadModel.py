from huggingface_hub import snapshot_download
import transformers
print(transformers.__version__)

# snapshot_download(repo_id="huggyllama/llama-13b",
#  local_dir="/remote-home1/yangshuo/N-LoRA/initial_model/llama-13b/",
#  local_dir_use_symlinks=False, max_workers=1 )


# T5-base
snapshot_download(repo_id="google/flan-t5-base",
 local_dir="initial_model/flan-t5-base",
 local_dir_use_symlinks=False, max_workers=1 )

# T5-large google-t5/t5-large


# T5-small 
# snapshot_download(repo_id="google/vit-base-patch16-224",
#  local_dir="initial_model/vit-base-patch16-224/",
#  local_dir_use_symlinks=False, max_workers=1 )






# vit-base-patch16-224-in21k
# snapshot_download(repo_id="google/vit-base-patch16-224-in21k",
#  local_dir="initial_model/vit-base-patch16-224-in21k/",
#  local_dir_use_symlinks=False, max_workers=1 )


#  openai/clip-vit-base-patch16
# snapshot_download(repo_id="openai/clip-vit-base-patch16",
#  local_dir="initial_model/clip-vit-base-patch16/",
#  local_dir_use_symlinks=False, max_workers=1 )
