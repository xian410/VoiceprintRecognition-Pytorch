# download_models_once.py
from modelscope import snapshot_download
import os

model_dir = "./models/asr"
os.makedirs(model_dir, exist_ok=True)

print("📥 下载 ASR 模型...")
snapshot_download(
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    cache_dir=model_dir,
)

print("📥 下载 VAD 模型...")
snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", cache_dir=model_dir)

print("📥 下载 标点模型...")
snapshot_download(
    "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", cache_dir=model_dir
)

print(f"✅ 所有模型已保存至: {os.path.abspath(model_dir)}")
