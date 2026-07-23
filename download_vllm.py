import urllib.request
url = "https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3/vllm/model_executor/layers/fused_moe/fused_moe.py"
urllib.request.urlretrieve(url, "vllm_fused_moe.py")
print("Downloaded to vllm_fused_moe.py")
