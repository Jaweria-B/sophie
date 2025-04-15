import asyncio
import openai
import pyttsx3

# Initialize TTS engine (global)
engine = pyttsx3.init()
engine.setProperty('rate', 185)

# Speak the buffered text using pyttsx3
def speak_text(text):
    engine.say(text)
    engine.runAndWait()
def stream_gpt4_response(command_text, callback):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": command_text}],
        max_tokens=150,
        stream=True
    )
    buffer = ""
    for chunk in response:
        content = chunk.choices[0].delta.get("content", "")
        buffer += content
        sentences, remaining = extract_sentences(buffer)
        for sentence in sentences:
            callback(sentence)  # Send to TTS
        buffer = remaining
    if buffer.strip():
        callback(buffer)  # Flush remaining text

import re

def extract_sentences(text):
    sentences = []
    remaining = text
    # Split on .!? followed by space or end-of-string
    matches = list(re.finditer(r'(?<=[.!?])\s+', text))
    if matches:
        split_idx = matches[-1].end()
        sentences = re.split(r'(?<=[.!?])\s+', text[:split_idx])
        remaining = text[split_idx:]
    return sentences, remaining

import edge_tts
import asyncio
from queue import Queue

tts_queue = Queue()

async def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None: break
        communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data = np.frombuffer(chunk["data"], dtype=np.int16)
                sd.play(audio_data, samplerate=24000)
                sd.wait()

def start_tts():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(tts_worker())

tts_thread = threading.Thread(target=start_tts, daemon=True)
tts_thread.start()