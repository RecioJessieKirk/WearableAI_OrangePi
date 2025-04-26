import whisper
import pyaudio
import numpy as np
import torch
import os
import time
import pyttsx3
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import wave
import io
import keyboard  # For real-time key press detection

# === TEXT-TO-SPEECH SETUP ===
engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    print(f"🗣️ TTS: {text}")
    engine.say(text)
    engine.runAndWait()

# === BEEP GENERATOR USING PYAUDIO ===
def play_beep(frequency=1000, duration=0.3, rate=44100):
    p = pyaudio.PyAudio()
    frames = int(rate * duration)
    t = np.linspace(0, duration, frames, False)
    tone = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)
    stream.write(tone.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# === LOAD LOCAL WHISPER MODEL ===
whisper_model = whisper.load_model("tiny", download_root="local_whisper_model")

# === LOAD LOCAL DISTILBERT MODEL ===
bert_path = "local_model/distilbert-base-nli-stsb-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
bert_model = AutoModel.from_pretrained(bert_path, local_files_only=True)

# === EMBEDDING FUNCTION ===
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted

# === PREDEFINED COMMANDS ===
commands = [
    "Open Object Detection",
    "Read the text",
    "exit"
]
command_embeddings = [get_embedding(cmd) for cmd in commands]
threshold = 0.6

# === MICROPHONE CONFIG ===
CHUNK = 1024
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.0  # seconds

# === RECORD UNTIL SILENCE ===
def record_until_silence():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0
    print("🎙️ Listening...")

    start_time = time.time()
    max_duration = 10  # seconds

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, np.int16)
        if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > (RATE / CHUNK * SILENCE_DURATION):
            break

        if time.time() - start_time > max_duration:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

# === TRANSCRIBE AUDIO ===
def transcribe(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
    return result.get("text", "").strip()

# === NLP COMMAND MATCHING ===
def match_command(text):
    user_embed = get_embedding(text)
    for i, cmd_embed in enumerate(command_embeddings):
        sim = F.cosine_similarity(user_embed, cmd_embed).item()
        if sim >= threshold:
            return commands[i], sim
    return None, None

# === MAIN LOOP ===
print("🤖 Ready. 🔵 Press and hold 'C' to speak. Say 'exit' to stop.")
try:
    while True:
        if keyboard.is_pressed('c'):
            time.sleep(0.2)  # debounce

            speak("Please speak after the tone.")
            play_beep()

            audio = record_until_silence()
            text = transcribe(audio)

            if not text:
                speak("I didn't detect any speech.")
                print("⚠️ No speech detected.")
                print("🔵 Press C to Speak")
                continue

            speak(f"You said: {text}")
            print(f"📝 Recognized Text: {text}")

            matched_cmd, sim = match_command(text.lower())

            if matched_cmd:
                speak(f"Matched command: {matched_cmd}")
                print(f"✅ Matched Command: {matched_cmd} (Similarity: {sim:.2f})")
                if matched_cmd == "exit":
                    speak("Exiting the program.")
                    break
            else:
                speak("Sorry, I didn't understand that.")
                print("❌ No matching command found.")

            print("🔵 Press C to Speak")  # Print reminder after every interaction

        time.sleep(0.1)  # avoid high CPU usage

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
