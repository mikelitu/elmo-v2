# transcribir lo q se dice + reproducir respuesta con emoción dinámica
import os
import json
from datetime import datetime
import ollama
import whisper
import speech_recognition as sr
import numpy as np
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

# 0. CONFIGURACIÓN GENERAL Y CLIENTES
API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_24df57d7e4df12509eb0c18ac689f059b418f469015e9d7b")
VOICE_ID = "V0wJiFHvBulVpU9dW2K1"
MODELO_OLLAMA = "neia8b" # modelo ollama a usar
client = ElevenLabs(api_key=API_KEY)

# CARPETA Y REGISTRO DE SESIONES JSON
CARPETA_SESIONES = "sesiones"
os.makedirs(CARPETA_SESIONES, exist_ok=True)
log_conversacion = []

def registrar_mensaje(emisor, texto):
    """Registra cada intervención en la lista de la sesión."""
    log_conversacion.append({emisor: texto})

# se guarda el .json al acabar la sesión en vez de durante para mayor rapidez procesando todo.
def guardar_sesion_json():
    """Transforma el historial de Ollama al formato NEIA/USER y lo guarda en JSON al terminar."""
    if not historial_mensajes:
        return

    log_conversacion = []
    for msg in historial_mensajes:
        emisor = "NEIA" if msg["role"] == "assistant" else "USER"
        log_conversacion.append({emisor: msg["content"]})

    ahora = datetime.now()
    nombre_archivo = ahora.strftime("session_[%d-%m-%Y][%H:%M].json")
    ruta_archivo = os.path.join(CARPETA_SESIONES, nombre_archivo)

    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(log_conversacion, f, ensure_ascii=False, indent=4)

    print(f"\n[Sesión guardada con éxito en: {ruta_archivo}]")

# CONFIGURACIÓN DE PARÁMETROS DE EMOCIÓN DE VOZ
PERFILES_VOZ = {
    "alegre": VoiceSettings(
        stability=0.22, # valor bajo = libertad IA = voz enérgica, dinámica...
        similarity_boost=0.75, # timbre característico de neia (sino suena mal)
        style=0.75, # valor alto = exagera matices positivos
        use_speaker_boost=True # limpiar la señal de audio
    ),
    "triste": VoiceSettings(
        stability=0.92,
        similarity_boost=0.75,
        style=0.00, # eliminar cualquier expresividad exagerada
        use_speaker_boost=True
    ),
    "regular": VoiceSettings(
        stability=0.50,
        similarity_boost=0.75,
        style=0.20,
        use_speaker_boost=True
    )
}

# estado de ánimo por defecto
ajustes_voz_actuales = PERFILES_VOZ["regular"]

# DICCIONARIO DE RESPUESTAS DIRECTAS A MONOSÍLABOS EN EL PASO INICIAL
CORTESIAS_INICIALES = {
    "si": "¡Dime! Pero cuéntame primero, ¿te encuentras bien, mal o regular?",
    "sí": "¡Dime! Pero cuéntame primero, ¿te encuentras bien, mal o regular?",
    "gracias": "De nada, Neia está aquí para ayudarte. Cuéntame, ¿cómo te sientes hoy?",
    "hola": "¡Hola! Cuéntame, ¿cómo te sientes hoy?"
}

def detectar_y_actualizar_emocion(texto):
    # detecta estado de ánimo en el texto usuario + actualiza parámetros de voz
    global ajustes_voz_actuales
    texto_l = texto.lower()

    # palabras estado = positivo
    if any(w in texto_l for w in ["bien", "genial", "alegre", "content", "estupend", "fenomenal", "excelente"]):
        ajustes_voz_actuales = PERFILES_VOZ["alegre"]
        print("[EMOCIÓN VOZ: Alegre / Animada]")

    # palabras estado = negativo
    elif any(w in texto_l for w in ["mal", "triste", "fatal", "deprimid", "horrible", "dolor", "asco"]):
        ajustes_voz_actuales = PERFILES_VOZ["triste"]
        print("[EMOCIÓN VOZ: Empática / Suave]")

    # palabras estado = neutro/regular
    elif any(w in texto_l for w in ["regular", "así así", "asi asi", "mas o menos", "más o menos", "cansad"]):
        ajustes_voz_actuales = PERFILES_VOZ["regular"]
        print("[EMOCIÓN VOZ: Intermedia / Serena]")

# 1. CARGA DE WHISPER
print("CARGANDO WHISPER MEDIUM...")
modelo_whisper = whisper.load_model("medium")

# 2. CONFIGURACIÓN DE MICRÓFONO
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("\nAJUSTANDO RUIDO DE FONDO...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
historial_mensajes = []

# SALUDO INICIAL DE NEIA
texto_saludo = "¡Hola! Soy Neia, ¿cómo te encuentras hoy?"
print(f"\nNEIA: {texto_saludo}\n")

audio_saludo = client.text_to_speech.convert(
    text=texto_saludo,
    voice_id=VOICE_ID,
    model_id="eleven_multilingual_v2",
    voice_settings=PERFILES_VOZ["regular"]
)
play(audio_saludo)

# guardar saludo en el historial de Ollama y en el JSON
historial_mensajes.append({"role": "assistant", "content": texto_saludo})
registrar_mensaje("NEIA", texto_saludo)

# BUCLE PRINCIPAL DE CONVERSACIÓN
try:
    while True:
        # 1. CAPTURA DE AUDIO
        with sr.Microphone() as source:
            print("\nESCUCHANDO...")
            audio = recognizer.listen(source)
        print("PROCESANDO AUDIO CON WHISPER...")
        # convertir audio a 16kHz float32 para Whisper
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        resultado = modelo_whisper.transcribe(audio_np, language="es")
        texto_usuario = resultado["text"].strip()

        if not texto_usuario:
            print("No se ha detectado de manera clara. Reintentando...")
            continue
        print(f"\nTRANSCRIPCIÓN: {texto_usuario}")

        # INTERCEPCIÓN DIRECTA DE MONOSÍLABOS EN EL PRIMER PASO
        texto_limpio = texto_usuario.lower().strip(".!?,")

        if len(historial_mensajes) <= 2 and texto_limpio in CORTESIAS_INICIALES:
            respuesta_filtro = CORTESIAS_INICIALES[texto_limpio]
            print(f"NEIA (Respuesta directa): {respuesta_filtro}\n")

            historial_mensajes.append({"role": "user", "content": texto_usuario})
            historial_mensajes.append({"role": "assistant", "content": respuesta_filtro})

            audio_respuesta = client.text_to_speech.convert(
                text=respuesta_filtro,
                voice_id=VOICE_ID,
                model_id="eleven_multilingual_v2",
                voice_settings=PERFILES_VOZ["regular"]
            )
            play(audio_respuesta)
            continue  # salta directamente al siguiente turno de escucha sin llamar a Ollama

        # EVALUAR Y ACTUALIZAR EMOCIÓN DE VOZ SEGÚN LO QUE HA DICHO EL USUARIO
        detectar_y_actualizar_emocion(texto_usuario)
        historial_mensajes.append({"role": "user", "content": texto_usuario})

        # registrar texto de usuario en JSON
        registrar_mensaje("USER", texto_usuario)

        # GENERACIÓN Y STREAMING POR FRASES CON LA NUEVA CONFIGURACIÓN DE VOZ
        print("NEIA: ", end="", flush=True)
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

            # al detectar un punto, signo o salto de línea => sintetiza esa frase
            if any(p in texto_chunk for p in ['.', '!', '?', '\n']):
                frase_para_hablar = buffer_frase.replace("[FIN]", "").strip()

                if es_inicio_respuesta:
                    if frase_para_hablar.lower().startswith("assistant"):
                        frase_para_hablar = frase_para_hablar[len("assistant"):].strip().lstrip(":")
                    es_inicio_respuesta = False

                if frase_para_hablar:
                    audio_chunk = client.text_to_speech.convert(
                        text=frase_para_hablar,
                        voice_id=VOICE_ID,
                        model_id="eleven_multilingual_v2",
                        voice_settings=ajustes_voz_actuales
                    )
                    play(audio_chunk)
                buffer_frase = ""

        # reproducir fragmentos pendientes al final
        frase_restante = buffer_frase.replace("[FIN]", "").strip()
        if frase_restante:
            if es_inicio_respuesta and frase_restante.lower().startswith("assistant"):
                frase_restante = frase_restante[len("assistant"):].strip().lstrip(":")
            if frase_restante:
                audio_chunk = client.text_to_speech.convert(
                    text=frase_restante,
                    voice_id=VOICE_ID,
                    model_id="eleven_multilingual_v2",
                    voice_settings=ajustes_voz_actuales
                )
                play(audio_chunk)
        print("\n")

        # guardar respuesta limpia en el historial
        respuesta_limpia = respuesta_completa.replace("[FIN]", "").strip()
        if respuesta_limpia.lower().startswith("assistant"):
            respuesta_limpia = respuesta_limpia[len("assistant"):].strip().lstrip(":")

        historial_mensajes.append({"role": "assistant", "content": respuesta_limpia})
        registrar_mensaje("NEIA", respuesta_limpia)

        if apagar_sistema:
            print("\n[Sistema apagado por comando de voz]")
            break

except KeyboardInterrupt:
    print("\n\nCONVERSACIÓN FINALIZADA POR EL USUARIO")

finally:
    guardar_sesion_json()
