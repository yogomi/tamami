"""Phase 0 PoC: uk / ja 音声での language="auto" 検証（FLEURS サンプル使用）.

言語タグ（<xx-XX>）の判定と認識テキストを、FLEURS の参照文と並べて確認する。
実行環境の構築手順と実測結果は DGX_SETUP.md の 0 章を参照。
"""

import io
from threading import Thread

import soundfile as sf
from datasets import Audio, load_dataset
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="cuda")
model.eval()
processor.set_num_lookahead_tokens(6)
sr = processor.feature_extractor.sampling_rate


def stream_transcribe_auto(audio):
    """language="auto" でストリーミング認識し、言語タグ込みの生テキストを返す.

    Args:
        audio: 16kHz mono の音声波形（1次元 float 配列）.

    Returns:
        special token（言語タグ等）を含む認識テキスト.
    """
    first_inputs = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="auto",
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    def gen():
        yield first_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel_idx = processor.num_mel_frames_first_audio_chunk
        hop = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start = mel_idx * hop - n_fft // 2
        while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            inputs = processor(
                audio[start:end],
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="auto",
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
            yield inputs.input_features
            mel_idx += processor.num_mel_frames_per_audio_chunk
            start = mel_idx * hop - n_fft // 2

    # 言語タグを見るため special token を残す
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=False)
    kwargs = {**first_inputs, "input_features": gen(), "streamer": streamer}
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    text = "".join(streamer)
    thread.join()
    return text


if __name__ == "__main__":
    for config, label in [("uk_ua", "ウクライナ語"), ("ja_jp", "日本語")]:
        print(f"\n=== {label} ({config}) ===")
        # datasets 4.x の音声デコードは torchcodec を要求するため、
        # raw バイトを soundfile でデコードする（依存を増やさない）
        ds = load_dataset("google/fleurs", config, split="validation", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, sample in enumerate(ds):
            if i >= 2:
                break
            audio, file_sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != sr:
                import librosa

                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            result = stream_transcribe_auto(audio)
            print(f"  [ref {i}] {sample['transcription'][:80]}")
            print(f"  [hyp {i}] {result.strip()[:100]}")
