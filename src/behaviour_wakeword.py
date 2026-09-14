import redis
import numpy as np
import noisereduce as nr
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
import time
import middleware as mw

# Audio configuration
CHUNK_SIZE = 1280    # 80ms chunk at 16kHz
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0  # 2 seconds of audio buffer context

# Buffer constants
BUFFER_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)  # 32,000 samples
CHUNK_FRAMES = BUFFER_SAMPLES // CHUNK_SIZE         # 25 frames

# Initialize Faster-Whisper Model
# Usamos el modelo "tiny" o "base" en CPU optimizado con int8 para baja latencia
# Si dispones de GPU NVIDIA, puedes cambiar a device="cuda", compute_type="float16"

print("Cargando modelo Faster-Whisper...")
whisper_model = WhisperModel("/home/idmind/elmo-v2/models/whisper-tiny", device="cpu", compute_type="int8")


class RealtimeNoiseFilter:
    def __init__(self, sample_rate=16000, chunk_size=1280):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        # Retain previous frame for overlapping window context
        self.prev_samples = np.zeros(self.chunk_size, dtype=np.float32)

    def clean_chunk(self, pcm_array: np.ndarray) -> np.ndarray:
        # Ensure input array matches expected chunk size
        if pcm_array.shape[0] != self.chunk_size:
            pcm_array = pcm_array[:self.chunk_size]

        samples = pcm_array.astype(np.float32)
        
        # Combine previous and current frame for smooth stationary filtering (2560 total)
        combined = np.concatenate([self.prev_samples, samples])
        
        # Apply stationary noise reduction
        cleaned = nr.reduce_noise(
            y=combined, 
            sr=self.sample_rate, 
            stationary=True, 
            prop_decrease=0.4
        )
        
        # Update state with current frame
        self.prev_samples = samples
        
        # Extract ONLY the last chunk_size (1280) elements
        chunk_cleaned = cleaned[-self.chunk_size:]
        return np.clip(chunk_cleaned, -32768, 32767).astype(np.int16)


def record_and_recognize_whisper(redis_client, denoiser, duration_sec=2.0):
    """
    Captura audio del stream de Redis durante `duration_sec` segundos 
    y lo transcribe con Faster-Whisper.
    """
    audio_chunks = []
    start_time = time.time()
    last_stream_id = '$'

    print(f"Escuchando voz con Faster-Whisper ({duration_sec}s)...")

    # 1. Grabar fragmentos durante 2 segundos
    while time.time() - start_time < duration_sec:
        response = redis_client.xread({'robot:mic_stream': last_stream_id}, block=500, count=1)
        if response:
            stream_name, messages = response[0]
            for message_id, data in messages:
                last_stream_id = message_id
                
                raw_pcm = data[b'pcm']
                pcm_array = np.frombuffer(raw_pcm, dtype=np.int16)
                
                # Filtrar ruido
                clean_pcm_array = denoiser.clean_chunk(pcm_array)
                
                # Convertir a float32 normalizado [-1.0, 1.0] que requiere Whisper
                pcm_float = clean_pcm_array.astype(np.float32) / 32768.0
                audio_chunks.append(pcm_float)

    if not audio_chunks:
        return ""

    # Concatenar todos los fragmentos en un array 1D de float32
    audio_data = np.concatenate(audio_chunks)

    # Prompt de contexto para priorizar nombres o palabras de presentación
    context_prompt = "Solo vas a recibir nombres. Nombres esperados: Mikel, Maitane, Lorea, Unai, Maialen, Estela, Rocio"

    # 2. Transcribir el buffer float32
    segments, info = whisper_model.transcribe(
        audio_data,
        language="es",
        beam_size=1,                   # 1 es más rápido para tiempo real
        initial_prompt=context_prompt, # Pistas de nombres
        vad_filter=False               # Desactivamos VAD interno porque ya grabamos 2s fijos
    )

    detected_text = " ".join([segment.text for segment in segments]).strip()
    return detected_text


# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

speech = mw.Speech()
speakers = mw.Speakers()
server = mw.Server()

# Initialize OpenWakeWord and Noise Filter
oww = WakeWordModel(models=["/home/idmind/elmo-v2/models/elmo.onnx"])
denoiser = RealtimeNoiseFilter(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)

# Initialize sliding window buffer for 2 seconds of audio (32,000 float32 samples)
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)

last_id = '$'  # Process only new incoming frames

print(f"Listening for wake word from Redis (2s sliding buffer = {BUFFER_SAMPLES} samples)...")

while True:

    # 1. Ignore incoming audio if the robot is currently speaking
    

    # Read incoming audio chunk from stream
    response = r.xread({'robot:mic_stream': last_id}, block=1000, count=1)
    
    if response:
        stream_name, messages = response[0]
        for message_id, data in messages:
            last_id = message_id
            
            raw_pcm = data[b'pcm']
            # Convert raw bytes back to Int16 numpy array
            pcm_array = np.frombuffer(raw_pcm, dtype=np.int16)

            # 1. Apply real-time denoiser to the 1280-sample frame
            clean_pcm_array = denoiser.clean_chunk(pcm_array)

            # 2. Normalize clean chunk to float32 range [-1.0, 1.0]
            pcm_float = clean_pcm_array.astype(np.float32) / 32768.0

            # 3. Slide new chunk into the 2-second audio buffer
            audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = pcm_float

            # 4. Predict using the full 2-second buffered window
            prediction = oww.predict(audio_buffer)
            
            if prediction.get("elmo", 0.0) >= 0.8:
                print("\n>>> WAKE WORD DETECTED! <<<")

                # Trigger speech output via Middleware
                # sound_url = server.url_for_sound("saludo.mp3")
                # speakers.url = sound_url
                
                # time.sleep(2.5)

                # Wait until it finishes speaking
                # print("Playing audio... blocking main script")
                # while speakers.playing is not None:
                #     time.sleep(0.05)
                
                # print("Playback complete! Resuming execution")

                # Escuchar y transcribir los siguientes 2 segundos con Faster-Whisper
                text_spoken = record_and_recognize_whisper(r, denoiser, duration_sec=2.0)
                print(f"Detected Speech (Whisper): '{text_spoken}'\n")

                # Clear audio buffer to prevent immediate double-triggering
                audio_buffer.fill(0.0)