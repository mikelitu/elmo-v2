# TRANSCRIBE LO QUE LE DICES Y REPRODUCE SU RESPUESTA SEGÚN LA INFO DE ELEVENLABS
import ollama
import whisper
import speech_recognition as sr
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

# 0. CONFIGURACIÓN CLIENTES
client = ElevenLabs (api_key="sk_62ed49c31c3c795b6d5c0e55475c333033492fc0128e7d66")

# 1. WHISPER
print("CARGANDO WHISPER MEDIUM")
modelo_whisper = whisper.load_model("medium")

# 2. MICRÓFONO (VAD Automático)
recognizer = sr.Recognizer()

# abrir micrófono + calibración sonido
with sr.Microphone() as source:
    print("\nAJUSTANDO RUIDO DE FONDO...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
historial_mensajes = []

# saludo inicial NAYE
texto_saludo = "¡Hola! Soy Naye, ¿cómo te encuentras hoy?"
audio_saludo = client.text_to_speech.convert(
  text=texto_saludo,
  voice_id="k14jZvkeJMVl6j0iG4G4",
  model_id="eleven_multilingual_v2"
)
play(audio_saludo)
# guardar saludo
historial_mensajes.append({"role": "assistant", "content": texto_saludo})

# bucle conversación hasta q ctrl+c
try:
  while True:
    # captura de audio
    with sr.Microphone() as source:
        print("\nESCUCHANDO")
        audio = recognizer.listen(source)
    print("PROCESANDO AUDIO")
    # transcripción a texto con whisper
    # obtener los datos puros ajustados a 16kHz (el estándar que exige Whisper)
    audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
    # datos en array matemático (Whisper puede leer directamente)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    # transcribir forzando el idioma a español para mayor velocidad
    resultado = modelo_whisper.transcribe(audio_np, language="es")
    texto_usuario = resultado["text"].strip()

    if not texto_usuario:
      print("No se ha detectado de manera clara. Reintentando...")
      continue
    print(f"\nTRANSCRIPCIÓN: {texto_usuario}")

    historial_mensajes.append({"role": "user", "content": texto_usuario})
    # ollama: llama3.2:1b + activar streaming
    respuesta = ollama.chat(model='naye', messages=historial_mensajes)
    respuesta_final = respuesta['message']['content']

    apagar_sistema = False
    if "[FIN]" in respuesta_final:
      apagar_sistema = True
      respuesta_final = respuesta_final.replace("[FIN]", "").strip()
    print(f"OLLAMA: {respuesta_final}\n")

    historial_mensajes.append(
      {"role": "assistant", "content": respuesta_final}
    )

    print("REPRODUCIENDO RESPUESTA")
    audio_respuesta = client.text_to_speech.convert(
      text=respuesta_final,
      voice_id="k14jZvkeJMVl6j0iG4G4",
      model_id="eleven_multilingual_v2",
      # output_format="mp3_44100_128",
    )
    play(audio_respuesta)
    
    if apagar_sistema:
      print("\n[Sistema apagado por comando de voz]")
      break

except KeyboardInterrupt:
  print("\n\n CONVERSACIÓN FINALIZADA POR EL USUARIO")