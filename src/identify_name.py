import redis
import numpy as np
import json
import time
from vosk import Model, KaldiRecognizer
import middleware as mw

# Audio configuration
CHUNK_SIZE = 1280    # 80ms chunk at 16kHz
SAMPLE_RATE = 16000

# 1. Initialize Vosk Model
print("Cargando modelo Vosk...")
vosk_model = Model("/home/idmind/elmo-v2/models/vosk-model-small-es-0.42")

# 2. Define custom grammar to restrict recognition solely to expected names + unknown token
with open("static/data/names.json", "r", encoding="utf-8") as f:
    data = json.load(f)

CUSTOM_NAMES = data["basque_names"] + data["common_spanish_names"]

grammar_json = json.dumps(CUSTOM_NAMES)

def recognize_name_vosk(redis_client, timeout_sec=3.0):
    """
    Reads continuous audio chunks from Redis and feeds them to Vosk 
    until a name is recognized or timeout is reached.
    """
    # Create a fresh recognizer instance for this listening turn
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE, grammar_json)
    
    start_time = time.time()
    last_stream_id = '$'  # Read only new incoming audio

    print(f"Escuchando nombre con Vosk (Timeout: {timeout_sec}s)...")

    while time.time() - start_time < timeout_sec:
        # Read incoming PCM chunk from Redis
        response = redis_client.xread({'robot:mic_stream': last_stream_id}, block=200, count=1)
        
        if response:
            stream_name, messages = response[0]
            for message_id, data in messages:
                last_stream_id = message_id
                
                raw_pcm = data[b'pcm']  # Raw 16-bit 16kHz PCM bytes
                
                # AcceptWaveform returns True when Vosk detects a pause/end of phrase
                if rec.AcceptWaveform(raw_pcm):
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip()
                    if text and text != "[unk]":
                        return text

    # Final check for partial result if timeout was reached mid-word
    final_res = json.loads(rec.FinalResult())
    text = final_res.get("text", "").strip()
    return text if text != "[unk]" else ""


# --- Example Integration Loop ---
r = redis.Redis(host='localhost', port=6379, db=0)
speakers = mw.Speakers()
server = mw.Server()

# Assume wake word or trigger event happens here:
print("Simulating trigger event...")

# Trigger greeting speech output
# speakers.url = server.url_for_sound("saludo.mp3")

# Wait until playback finishes before opening the microphone loop
time.sleep(0.15)
while speakers.playing is not None:
    time.sleep(0.05)

# Fast streaming recognition via Vosk
detected_name = recognize_name_vosk(r, timeout_sec=2.5)
print(f"Detected Name (Vosk): '{detected_name}'")