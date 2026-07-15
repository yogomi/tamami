"""Phase 0 PoC: Transformers 経路のストリーミング推論を実時間ペーシングで実測する.

チャンクを音声の実時間どおりに供給し、以下を測る:
- 最初の暫定テキストが「対応する音声の発話時点」からどれだけ遅れて出るか
- チャンクあたりの推論時間（実時間比 RTF）
- 全チャンク供給後、最終テキストが出きるまでの時間

実行環境の構築手順と実測結果は DGX_SETUP.md の 0 章を参照。
"""

import time
from threading import Thread

from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer
from transformers.audio_utils import load_audio

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map="cuda")
model.eval()

processor.set_num_lookahead_tokens(6)
sr = processor.feature_extractor.sampling_rate
print(f"streaming latency (nominal): {processor.streaming_latency_ms} ms")
print(f"supported latencies: {processor.supported_streaming_latencies_ms}")
print(
    f"samples/chunk: {processor.num_samples_per_audio_chunk} "
    f"({processor.num_samples_per_audio_chunk / sr * 1000:.0f} ms)"
)


def stream_transcribe(audio, language, paced=True):
    """実時間ペーシングでストリーミング認識し、遅延メトリクスを表示する.

    Args:
        audio: 16kHz mono の音声波形（1次元 float 配列）.
        language: 言語プロンプト（"en-US" 等のロケール、または "auto"）.
        paced: True なら各チャンクを音声の実時間どおりに供給する.
            False なら計算速度の上限（RTF）を測る.

    Returns:
        認識されたテキスト全体.
    """
    first_inputs = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    t_start = time.perf_counter()
    chunk_infer_times = []
    feed_done_at = [None]

    def gen():
        t_prev = time.perf_counter()
        first_chunk_sec = processor.num_samples_first_audio_chunk / sr
        if paced:
            time.sleep(max(0.0, t_start + first_chunk_sec - time.perf_counter()))
        yield first_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

        mel_idx = processor.num_mel_frames_first_audio_chunk
        hop = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start = mel_idx * hop - n_fft // 2
        while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            t_before = time.perf_counter()
            chunk_infer_times.append(t_before - t_prev)
            if paced:
                # このチャンク末尾の音声が「発話され終わる」実時刻まで待つ
                time.sleep(max(0.0, t_start + end / sr - time.perf_counter()))
            inputs = processor(
                audio[start:end],
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=language,
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
            t_prev = time.perf_counter()
            yield inputs.input_features
            mel_idx += processor.num_mel_frames_per_audio_chunk
            start = mel_idx * hop - n_fft // 2
        feed_done_at[0] = time.perf_counter() - t_start

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    kwargs = {**first_inputs, "input_features": gen(), "streamer": streamer}
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    events = []  # (経過秒, 累積テキスト)
    text = ""
    for piece in streamer:
        text += piece
        if piece.strip():
            events.append((time.perf_counter() - t_start, text))
    thread.join()
    t_end = time.perf_counter() - t_start

    audio_sec = audio.shape[0] / sr
    print(f"  audio: {audio_sec:.1f}s | total wall: {t_end:.1f}s | paced: {paced}")
    if events:
        print(f"  first text at: {events[0][0]:.2f}s -> {events[0][1]!r}")
        print(f"  final text after audio end: {t_end - (feed_done_at[0] or audio_sec):.2f}s")
    if chunk_infer_times:
        worst = max(chunk_infer_times[1:] or chunk_infer_times)
        print(
            f"  chunk infer: worst {worst * 1000:.0f}ms "
            f"(chunk={processor.num_samples_per_audio_chunk / sr * 1000:.0f}ms)"
        )
    print(f"  text: {text.strip()[:120]!r}")
    return text


if __name__ == "__main__":
    print("\n=== English sample (language=en-US, paced) ===")
    audio = load_audio(
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples"
        "/resolve/main/obama.mp3",
        sampling_rate=sr,
    )
    stream_transcribe(audio, "en-US", paced=True)

    print("\n=== English sample (language=auto, unpaced = 純粋な計算速度) ===")
    stream_transcribe(audio, "auto", paced=False)
