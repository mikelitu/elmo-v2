import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from speechbrain.lobes.features import Fbank
import pyaudio

# 1. Initialize SpeechBrain Fbank Feature Extractor (Matches Model Training)
feature_extractor = Fbank(n_mels=80)

# 2. Load ONNX Model Session
session = ort.InferenceSession("ecapa_tdnn_encoder.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name


def pcm_to_mel_features(pcm_data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Converts raw 16-bit PCM audio bytes from a microphone 
    into an (1, time_frames, 80) Mel-Filterbank feature array.
    """
    # Step A: Convert raw bytes to Float32 Tensor [-1.0, 1.0]
    audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Convert to PyTorch Tensor [1, num_samples]
    waveform = torch.from_numpy(audio_float32).unsqueeze(0)
    
    # Resample if microphone rate isn't 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Step B: Extract 80-bin Mel Filterbanks using SpeechBrain's Fbank module
    with torch.no_grad():
        # Output shape: [1, time_frames, 80]
        mel_features = feature_extractor(waveform)

    return mel_features.numpy().astype(np.float32)

def extract_speaker_embedding(pcm_data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Extracts a normalized 192-d voice vector from raw mic recording."""
    # 1. Adapt audio to model input shape
    mel_input = pcm_to_mel_features(pcm_data, sample_rate)
    
    # 2. Pass features to ONNX model
    onnx_outputs = session.run(None, {input_name: mel_input})[0]
    
    # 3. Squeeze vector and apply L2 Normalization (for Cosine Similarity)
    embedding = onnx_outputs.squeeze()
    normalized_embedding = embedding / np.linalg.norm(embedding)
    
    return normalized_embedding


# Microphone Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3
save_embedding = False  # Set to True to save the embedding for later comparison
compare_embedding = True  # Set to True to compare with a saved embedding

p = pyaudio.PyAudio()

print("Recording audio for voice identification...")
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")
stream.stop_stream()
stream.close()
p.terminate()

# Combine recorded chunks into a single byte stream
pcm_bytes = b''.join(frames)

# Extract voiceprint embedding
embedding = extract_speaker_embedding(pcm_bytes, sample_rate=RATE)
print(f"Generated Voice Embedding Vector Shape: {embedding.shape}")  # (192,)

if save_embedding:
# Example: Save embedding to a file for later comparison
    np.save("voice_embedding.npy", embedding)

if compare_embedding:
    # Example: Load a previously saved embedding for comparison
    try:
        saved_embedding = np.load("voice_embedding.npy")
        similarity = np.dot(embedding, saved_embedding)  # Cosine similarity since both are normalized
        print(f"Cosine Similarity with Saved Embedding: {similarity:.4f}")
    except FileNotFoundError:
        print("No saved embedding found for comparison.")