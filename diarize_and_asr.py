import os
import argparse
import functools
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

# ====== 1. 说话人日志模块（你的现有代码） ======
from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

# ====== 2. ASR 模块（以 FunASR 为例） ======
from funasr import AutoModel


def init_asr_model():
    import torch
    from funasr import AutoModel
    import os

    base_dir = "./models/asr/iic"

    asr_path = os.path.join(
        base_dir, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    vad_path = os.path.join(base_dir, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    punc_path = os.path.join(
        base_dir, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    )

    # 确保路径存在
    for name, path in [("ASR", asr_path), ("VAD", vad_path), ("PUNC", punc_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} 模型未找到，请先运行 download_models_once.py"
            )

    print("正在加载本地 ASR 模型...")
    model = AutoModel(
        model=asr_path,  # 👈 本地路径
        vad_model=vad_path,  # 👈 本地路径
        punc_model=punc_path,  # 👈 本地路径
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,
    )
    return model


def transcribe_audio(asr_model, audio_path):
    """对单个音频文件进行 ASR 识别"""
    try:
        result = asr_model.generate(input=audio_path)
        text = result[0]["text"] if result and "text" in result[0] else ""
        return text.strip()
    except Exception as e:
        print(f"ASR 识别失败 {audio_path}: {e}")
        return ""


# ====== 3. 音频切片工具 ======
def slice_audio(audio_path, segments, output_dir):
    """
    按时间段切片音频

    Args:
        audio_path: 原始音频路径
        segments: [{"start": s, "end": e, "speaker": spk}, ...]
        output_dir: 切片保存目录

    Returns:
        list of dict with 'speaker', 'slice_path', 'start', 'end'
    """
    # 加载音频
    audio = AudioSegment.from_file(audio_path)
    sr = audio.frame_rate

    os.makedirs(output_dir, exist_ok=True)
    sliced_segments = []

    for i, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        slice_audio = audio[start_ms:end_ms]

        slice_path = os.path.join(output_dir, f"seg_{i:04d}_{seg['speaker']}.wav")
        slice_audio.export(slice_path, format="wav")
        sliced_segments.append(
            {
                "speaker": seg["speaker"],
                "slice_path": slice_path,
                "start": seg["start"],
                "end": seg["end"],
            }
        )
    return sliced_segments


# ====== 4. 主流程 ======
def main(args):
    # Step 1: 说话人日志
    predictor = MVectorPredictor(
        configs=args.configs,
        model_path=args.model_path,
        threshold=args.threshold,
        audio_db_path=args.audio_db_path,
        use_gpu=args.use_gpu,
    )

    results = predictor.speaker_diarization(
        args.audio_path,
        speaker_num=args.speaker_num,
        search_audio_db=True,  # 日志阶段不需要搜索库
    )

    # 转为标准格式（results 是 dict 列表）
    segments = []
    unknown_clusters = {}
    unknown_counter = 0

    for res in results:
        spk_label = res["speaker"]

        # 判断是否为已知注册说话人（策略可根据实际调整）
        is_known = False
        if isinstance(spk_label, str):
            # 假设注册名不会是 "陌生人X" 或纯数字
            if (
                not spk_label.startswith(("陌生人", "unknown", "Unknown"))
                and not spk_label.isdigit()
            ):
                is_known = True

        if is_known:
            final_spk = spk_label
        else:
            # 使用原始 label 作为 cluster key（支持 int 或 str）
            key = spk_label
            if key not in unknown_clusters:
                unknown_clusters[key] = f"说话人{unknown_counter}"
                unknown_counter += 1
            final_spk = unknown_clusters[key]

        segments.append(
            {
                "start": float(res["start"]),
                "end": float(res["end"]),
                "speaker": final_spk,
            }
        )

    print(
        f"共检测到 {len(segments)} 个语音片段，包含 {len(set(s['speaker'] for s in segments))} 位说话人"
    )

    # Step 2: 切片
    slice_dir = os.path.join("output", "slices")
    sliced_segments = slice_audio(args.audio_path, segments, slice_dir)

    # Step 3: 初始化 ASR
    print("正在加载 ASR 模型...")
    asr_model = init_asr_model()

    # Step 4: 逐片段识别
    final_output = {}
    for item in sliced_segments:
        # 确保 speaker 是字符串（或原生 int）
        spk = str(item["speaker"])  # 👈 关键：转为 str
        text = transcribe_audio(asr_model, item["slice_path"])
        if text:
            if spk not in final_output:
                final_output[spk] = []
            final_output[spk].append(
                {"start": item["start"], "end": item["end"], "text": text}
            )
        print(f"[{spk}] {text}")

    # Step 5: 保存结果
    os.makedirs("output", exist_ok=True)

    # JSON 格式（结构化）
    with open("output/transcript.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    # TXT 格式（可读性强）
    with open("output/transcript.txt", "w", encoding="utf-8") as f:
        for spk in sorted(final_output.keys()):
            f.write(f"\n【{spk}】\n")
            for utt in final_output[spk]:
                f.write(f"[{utt['start']:.1f}s - {utt['end']:.1f}s] {utt['text']}\n")

    print("\n✅ 处理完成！结果已保存至 output/ 目录")
    print("  - transcript.json: 结构化数据")
    print("  - transcript.txt: 可读文本")


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("configs", str, "configs/cam++.yml", "配置文件")
    add_arg("audio_path", str, "dataset/test_long.wav", "输入音频路径")
    add_arg("audio_db_path", str, "audio_db/", "音频库路径（可选）")
    add_arg("speaker_num", int, None, "说话人数量（可选）")
    add_arg("use_gpu", bool, True, "是否使用GPU")
    add_arg("threshold", float, 0.6, "说话人相似度阈值")
    add_arg("model_path", str, "models/CAMPPlus_Fbank/best_model/", "说话人模型路径")
    args = parser.parse_args()

    main(args)
