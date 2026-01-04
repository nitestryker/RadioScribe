import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SR = 16000
CHUNK_SEC = 4
OVERLAP_SEC = 1

MODEL_ID = "Systran/faster-distil-whisper-large-v3"

def pcm16_to_float32(x):
    return (x.astype(np.float32) / 32768.0).copy()

def main():
    model = WhisperModel(MODEL_ID, device="cpu", compute_type="int8")
    print("[asr] ready")

    buffer = np.zeros(0, dtype=np.float32)

    while True:
        print("[mic] listening...")
        audio = sd.rec(int(CHUNK_SEC * SR), samplerate=SR, channels=1, dtype="int16")
        sd.wait()

        f32 = pcm16_to_float32(audio.reshape(-1))
        buffer = np.concatenate([buffer, f32])

        if len(buffer) < SR * CHUNK_SEC:
            continue

        segment = buffer[-int((CHUNK_SEC + OVERLAP_SEC) * SR):]

        segments, _ = model.transcribe(
            segment,
            language="en",
            vad_filter=True,
            beam_size=1,
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            print(">>", text)

        time.sleep(0.1)

if __name__ == "__main__":
    main()
