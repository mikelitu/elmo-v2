import asyncio
import math
import struct
import numpy as np
import pyaudio
from livekit.wakeword import WakeWordModel

MODEL_PATH = "./output/elmo/elmo.onnx"
CHUNK_SIZE = 1280    # 80ms chunk at 16kHz
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0  # Total length of the audio context buffer in seconds

# Total number of raw audio samples for 2 seconds (32,000 samples)
BUFFER_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)

# Number of 80ms frames that make up the buffer (25 frames)
CHUNK_FRAMES = BUFFER_SAMPLES // CHUNK_SIZE

def get_audio_rms(pcm_bytes: bytes) -> float:
    """Calculates volume level (RMS) from raw 16-bit PCM bytes."""
    if not pcm_bytes:
        return 0.0
    count = len(pcm_bytes) // 2
    shorts = struct.unpack(f"{count}h", pcm_bytes)
    sum_squares = sum(s ** 2 for s in shorts)
    return math.sqrt(sum_squares / count)

async def main():
    print("Loading ONNX model...")
    model = WakeWordModel(models=[MODEL_PATH])

    p = pyaudio.PyAudio()
    stream = p.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("\n==============================================")
    print(" [STATUS] Microphone is OPEN & LISTENING ")
    print(" Speak into the mic to see predictions live!")
    print(f" Buffer size: {BUFFER_SAMPLES} samples ({CHUNK_FRAMES} frames)")
    print("==============================================\n")

    # Sliding buffer to maintain 2 seconds of temporal context (32,000 samples)
    audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)

    try:
        while True:
            # 1. Read PCM frame from microphone
            raw_pcm = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # 2. Check volume
            rms = get_audio_rms(raw_pcm)
            volume_bar = "|" * int(min(rms / 100, 20))

            # 3. Convert raw bytes to normalized float32 array [-1.0, 1.0]
            pcm_chunk = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

            # 4. Update sliding window buffer with new 1280-sample chunk
            audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = pcm_chunk

            # 5. Predict using the full 2-second buffered audio window
            scores = model.predict(audio_buffer)

            # Get highest confidence score from predictions dictionary
            max_score = 0.0
            detected_name = ""
            for name, score in scores.items():
                if score > max_score:
                    max_score = score
                    detected_name = name

            # Live terminal update on a single line
            print(f"\rListening... Vol: [{volume_bar:<20}] | Score ({detected_name}): {max_score:.2%}", end="", flush=True)

            # Trigger notification
            if max_score >= 0.8:
                print(f"\n\n>>> WAKE WORD DETECTED: {detected_name} ({max_score:.2%}) <<<\n")

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping listener...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())