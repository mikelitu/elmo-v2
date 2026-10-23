# TRANSCRIBE LO QUE LE DICES Y REPRODUCE SU RESPUESTA EN TIEMPO REAL (STREAMING + FILTRO DE CORTESÍA)
import ollama
import whisper
import speech_recognition as sr
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

# 0. CONFIGURACIÓN CLIENTES
client = ElevenLabs(api_key="ELEVENLABS_API_KEY")

# MODELO OLLAMA A USAR
MODELO_OLLAMA = "naye8b"

# DICCIONARIO DE RESPUESTAS DIRECTAS A MONOSÍLABOS EN EL PASO INICIAL
CORTESIAS_INICIALES = {
    "si": "¡Dime! Pero cuéntame primero, ¿te encuentras bien, mal o regular?",
    "sí": "¡Dime! Pero cuéntame primero, ¿te encuentras bien, mal o regular?",
    "gracias": "De nada, Naye está aquí para ayudarte. Cuéntame, ¿cómo te sientes hoy?",
    "hola": "¡Hola! Cuéntame, ¿cómo te sientes hoy?"
}

# 1. CARGA DE WHISPER
print("CARGANDO WHISPER MEDIUM...")
modelo_whisper = whisper.load_model("medium")

# 2. CONFIGURACIÓN DE MICRÓFONO
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("\nAJUSTANDO RUIDO DE FONDO...")
    recognizer.adjust_for_ambient_noise(source, duration=1)

historial_mensajes = []

# SALUDO INICIAL DE NAYE
texto_saludo = "¡Hola! Soy Naye, ¿cómo te encuentras hoy?"
print(f"\nNAYE: {texto_saludo}\n")

audio_saludo = client.text_to_speech.convert(
    text=texto_saludo,
    voice_id="ID_VOICE_ELEVENLABS",
    model_id="eleven_multilingual_v2"
)
play(audio_saludo)
# guardar saludo en el historial
historial_mensajes.append({"role": "assistant", "content": texto_saludo})

# BUCLE PRINCIPAL DE CONVERSACIÓN
try:
    while True:
        # 1. CAPTURA DE AUDIO
        with sr.Microphone() as source:
            print("\nESCUCHANDO...")
            audio = recognizer.listen(source)
        print("PROCESANDO AUDIO CON WHISPER...")
        # Convertir audio a 16kHz float32 para Whisper
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        resultado = modelo_whisper.transcribe(audio_np, language="es")
        texto_usuario = resultado["text"].strip()

        if not texto_usuario:
            print("No se ha detectado de manera clara. Reintentando...")
            continue

        print(f"\nTRANSCRIPCIÓN: {texto_usuario}")

        # INTERCEPCIÓN DIRECTA DE CORTESÍAS/MONOSÍLABOS EN EL PRIMER PASO
        texto_limpio = texto_usuario.lower().strip(".!?,")

        if len(historial_mensajes) <= 2 and texto_limpio in CORTESIAS_INICIALES:
            respuesta_filtro = CORTESIAS_INICIALES[texto_limpio]
            print(f"NAYE (Respuesta directa): {respuesta_filtro}\n")

            historial_mensajes.append({"role": "user", "content": texto_usuario})
            historial_mensajes.append({"role": "assistant", "content": respuesta_filtro})

            audio_respuesta = client.text_to_speech.convert(
                text=respuesta_filtro,
                voice_id="ID_VOICE_ELEVENLABS",
                model_id="eleven_multilingual_v2",
            )
            play(audio_respuesta)
            continue  # Salta directamente al siguiente turno de escucha sin llamar a Ollama

        # SI NO ES CORTESÍA INICIAL, AÑADIR AL HISTORIAL Y PROCESAR EN OLLAMA
        historial_mensajes.append({"role": "user", "content": texto_usuario})

        # 2. GENERACIÓN Y REPRODUCCIÓN EN STREAMING (POR FRASES)
        print("NAYE: ", end="", flush=True)

        stream_ollama = ollama.chat(
            model=MODELO_OLLAMA,
            messages=historial_mensajes,
            stream=True
        )

        buffer_frase = ""
        respuesta_completa = ""
        es_inicio_respuesta = True
        apagar_sistema = False

        for chunk in stream_ollama:
            texto_chunk = chunk['message']['content']
            respuesta_completa += texto_chunk
            buffer_frase += texto_chunk
            print(texto_chunk, end="", flush=True)

            if "[FIN]" in respuesta_completa:
                apagar_sistema = True

            # En cuanto detecta un punto, signo o salto de línea, sintetiza esa frase
            if any(p in texto_chunk for p in ['.', '!', '?', '\n']):
                frase_para_hablar = buffer_frase.replace("[FIN]", "").strip()

                if es_inicio_respuesta:
                    if frase_para_hablar.lower().startswith("assistant"):
                        frase_para_hablar = frase_para_hablar[len("assistant"):].strip().lstrip(":")
                    es_inicio_respuesta = False

                if frase_para_hablar:
                    audio_chunk = client.text_to_speech.convert(
                        text=frase_para_hablar,
                        voice_id="V0wJiFHvBulVpU9dW2K1",
                        model_id="eleven_multilingual_v2",
                    )
                    play(audio_chunk)

                buffer_frase = ""

        # Reproducir fragmentos pendientes al final
        frase_restante = buffer_frase.replace("[FIN]", "").strip()
        if frase_restante:
            if es_inicio_respuesta and frase_restante.lower().startswith("assistant"):
                frase_restante = frase_restante[len("assistant"):].strip().lstrip(":")
            if frase_restante:
                audio_chunk = client.text_to_speech.convert(
                    text=frase_restante,
                    voice_id="V0wJiFHvBulVpU9dW2K1",
                    model_id="eleven_multilingual_v2",
                )
                play(audio_chunk)

        print("\n")

        # Guardar respuesta limpia en el historial
        respuesta_limpia = respuesta_completa.replace("[FIN]", "").strip()
        if respuesta_limpia.lower().startswith("assistant"):
            respuesta_limpia = respuesta_limpia[len("assistant"):].strip().lstrip(":")

        historial_mensajes.append({"role": "assistant", "content": respuesta_limpia})

        if apagar_sistema:
            print("\n[Sistema apagado por comando de voz]")
            break

except KeyboardInterrupt:
    print("\n\nCONVERSACIÓN FINALIZADA POR EL USUARIO")
