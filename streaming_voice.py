import openai
import pyttsx3
import threading
from queue import Queue
import re

# Global TTS engine setup
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 180)
tts_engine.setProperty("voice", "english")  # Force English voice

# Thread-safe queue for sentences
tts_queue = Queue()
is_speaking = False

def tts_worker():
    global is_speaking
    while True:
        text = tts_queue.get()
        if text is None:
            break
        is_speaking = True
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        is_speaking = False


# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Modify the stream_gpt4_response function
def stream_gpt4_response(command_text, callback):
    response = openai.ChatCompletion.create(
        # Old
        # model="gpt-4",
        # New
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": command_text}],
        max_tokens=1000,
        stream=True
    )
    buffer = ""
    for chunk in response:
        if content := chunk.choices[0].delta.get("content", ""):
            buffer += content
            # Split on sentence boundaries
            while True:
                match = re.search(r'[.!?]\s+', buffer)
                if not match:
                    break
                split_pos = match.end()
                sentence = buffer[:split_pos].strip()
                if sentence:
                    callback(sentence)
                    tts_queue.put(sentence)  # Directly add to queue
                buffer = buffer[split_pos:]
    if buffer.strip():
        callback(buffer)
        tts_queue.put(buffer)