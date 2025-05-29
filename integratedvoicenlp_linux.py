import whisper
import pyaudio
import numpy as np
import torch
import os
import time
import gc
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import subprocess
import sys
from pynput import keyboard as pynput_keyboard

from tts_handler import speak

print(f"[INFO] NLP started — PID: {os.getpid()}")

# === MICROPHONE CONFIG ===
CHUNK = 1024
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.0  # seconds

def cleanup_model(model):
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Cleanup Error] {e}")

def load_whisper():
    return whisper.load_model("tiny", download_root="local_whisper_model")

def record_until_silence():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    silent_chunks = 0
    print("[MIC] Listening...")

    start_time = time.time()
    max_duration = 10
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

def transcribe(audio_bytes, model):
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="en", verbose=False)
    return result.get("text", "").strip()

# Improved command mapping with aliases
command_map = {
    "Open Object Detection": ["open object detection", "what is in front of me", "what's in front of me"],
    "Read the text": ["read the text"],
    "Open Pothole Detection": ["pothole detection", "detect pothole", "detect road damage"],
    "Check Banknote": ["how much money", "count the money", "money detector", "scan bills", "detect banknotes", "total money", "identify money"],
    "Exit": ["exit"]
}
threshold = 0.6

def get_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted

def prepare_alias_embeddings(tokenizer, bert_model):
    alias_to_command = {}
    alias_embeddings = []
    for command, aliases in command_map.items():
        for alias in aliases:
            alias_to_command[alias] = command
            alias_embeddings.append((alias, get_embedding(alias, tokenizer, bert_model)))
    return alias_to_command, alias_embeddings

def match_command(text, alias_to_command, alias_embeddings):
    user_embed = get_embedding(text, tokenizer, bert_model)
    for alias, alias_embed in alias_embeddings:
        sim = F.cosine_similarity(user_embed, alias_embed).item()
        if sim >= threshold:
            return alias_to_command[alias], sim
    return None, None

# Using pynput to detect 'c' key press & hold
c_key_pressed = False

def on_press(key):
    global c_key_pressed
    try:
        if key.char == 'c':
            c_key_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global c_key_pressed
    try:
        if key.char == 'c':
            c_key_pressed = False
    except AttributeError:
        pass

listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def run_script_and_exit(script_name):
    message_map = {
        "cv_yolov5_pt_linux.py": "Loading Object Identifier. Please wait.",
        "cv_ocr_linux.py": "Loading Text Reader. Please wait.",
        "cv_yolo_pothole_linux.py": "Loading Road Inspector. Please wait.",
        "cv_yolo_banknote_linux.py": "Loading Money Counter. Please wait."
    }
    speak(message_map.get(script_name, f"Opening {script_name}. Please wait."))

    listener.stop()
    cleanup_model(whisper_model)
    globals()["whisper_model"] = None

    subprocess.call([sys.executable, script_name])
    sys.exit(0)

def open_object_detection():
    run_script_and_exit("cv_yolov5_pt_linux.py")

def open_ocr():
    run_script_and_exit("cv_ocr_linux.py")

def open_pothole_detection():
    run_script_and_exit("cv_yolo_pothole_linux.py")

def open_banknote_detector():
    run_script_and_exit("cv_yolo_banknote_linux.py")

def play_beep():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)
    beep = (0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, 0.3, int(44100 * 0.3)))).astype(np.float32)
    stream.write(beep.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    global whisper_model, tokenizer, bert_model

    whisper_model = load_whisper()

    bert_path = "local_model/distilbert-base-nli-stsb-mean-tokens"
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    bert_model = AutoModel.from_pretrained(bert_path, local_files_only=True)

    alias_to_command, alias_embeddings = prepare_alias_embeddings(tokenizer, bert_model)

    startup_message = "Ready. Press and hold the Button to speak. Say exit to stop."
    print(startup_message)
    speak(startup_message)

    try:
        while True:
            if c_key_pressed:
                # Debounce & prompt
                time.sleep(0.2)
                speak("Please speak after the tone.")
                play_beep()

                audio = record_until_silence()
                text = transcribe(audio, whisper_model)
                if not text:
                    speak("I didn't detect any speech.")
                    print("No speech detected.")
                    print("Press the Button to speak")
                    continue

                speak(f"You said: {text}")
                print(f"Recognized Text: {text}")

                matched_cmd, sim = match_command(text.lower(), alias_to_command, alias_embeddings)
                if matched_cmd:
                    print(f"Matched Command: {matched_cmd} (Similarity: {sim:.2f})")
                    if matched_cmd == "Open Object Detection":
                        open_object_detection()
                    elif matched_cmd == "Read the text":
                        open_ocr()
                    elif matched_cmd == "Open Pothole Detection":
                        open_pothole_detection()
                    elif matched_cmd == "Check Banknote":
                        open_banknote_detector()
                    elif matched_cmd == "Exit":
                        speak("Exiting the program.")
                        listener.stop()
                        break
                else:
                    speak("Sorry, I didn't understand that.")
                    print("No matching command found.")
                print("Press the Button to speak")
            time.sleep(0.1)

    except KeyboardInterrupt:
        listener.stop()
        print("\nStopped by user.")
    finally:
        cleanup_model(whisper_model)
        cleanup_model(bert_model)

if __name__ == "__main__":
    main()
