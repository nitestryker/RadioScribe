import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SECONDS = 60

MODEL_ID = "Systran/faster-distil-whisper-large-v3"  # Option A
DEVICE = "cpu"   # change to "cuda" later if you want
COMPUTE_TYPE = "int8"  # good CPU default; try "int8_float16" or "float16" on GPU

def record_pcm16(seconds: int, sr: int) -> np.ndarray:
    print(f"[mic] recording {seconds}s @ {sr}Hz ...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)

def pcm16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).copy()

def main():
    print("[asr] loading model...")
    model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("[asr] model loaded.")

    # quick warmup
    time.sleep(0.2)

    pcm16 = record_pcm16(SECONDS, SAMPLE_RATE)
    audio = pcm16_to_float32(pcm16)

    print("[asr] transcribing...")
    segments, info = model.transcribe(
        audio,
        language="en",
        vad_filter=True,      # helps a lot for radio/noisy input
        beam_size=1,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    print("\n--- RESULT ---")
    print(text if text else "(no speech detected)")
    print("--------------")

if __name__ == "__main__":
    main()
