import openai
import pyttsx3
import threading
import re
from queue import Queue
from src.prompts.prompts import assistant_prompt, RAG_SEARCH_PROMPT_TEMPLATE  

# Global TTS engine setup
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 180)
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[0].id)  # Use first English voice

EMOTION_SETTINGS = {
    "thoughtful": {"rate": 150, "volume": 0.9},
    "warm": {"rate": 180, "volume": 1.0},
    "playful": {"rate": 220, "volume": 1.0},
    "sarcastic": {"rate": 160, "volume": 0.95},
    "mischievous": {"rate": 210, "volume": 0.95},
    "annoyed": {"rate": 140, "volume": 1.1},
    "enthusiastic": {"rate": 240, "volume": 1.0},
    "frustrated": {"rate": 130, "volume": 1.05},
    "skeptical": {"rate": 170, "volume": 0.9},
    "curious": {"rate": 190, "volume": 1.0},
    "neutral": {"rate": 180, "volume": 1.0}
}

EMOTION_TAG_PATTERN = re.compile(r'\[EMOTION\](.*?)\[/EMOTION\]')

# Thread-safe queue for sentences
tts_queue = Queue()
is_speaking = False

def tts_worker():
    global is_speaking
    while True:
        text = tts_queue.get()
        if text is None:
            break
        
        # Extract emotion from text
        emotion, _, clean_text = text.partition(":")
        emotion = emotion.strip().lower()
        
        # Apply emotion settings
        settings = EMOTION_SETTINGS.get(emotion, {"rate": 180, "volume": 1.0})
        tts_engine.setProperty("rate", settings["rate"])
        tts_engine.setProperty("volume", settings["volume"])
        
        is_speaking = True
        try:
            tts_engine.say(clean_text.strip())
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        is_speaking = False

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def stream_gpt4_response(command_text, callback):
    # Combine system prompt with task-specific instructions
    full_system_prompt = f"{assistant_prompt}\n\n{RAG_SEARCH_PROMPT_TEMPLATE}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": command_text}
        ],
        max_tokens=1000,
        stream=True
    )
    
    buffer = ""
    current_emotion = "neutral"
    
    for chunk in response:
        if content := chunk.choices[0].delta.get("content", ""):
            buffer += content
            
            # Extract and process emotion tags
            emotion_match = EMOTION_TAG_PATTERN.search(buffer)
            if emotion_match:
                current_emotion = emotion_match.group(1).strip().lower()
                buffer = EMOTION_TAG_PATTERN.sub('', buffer)  # Remove tags from text

            # Split on sentence boundaries
            while True:
                match = re.search(r'[.!?]\s+', buffer)
                if not match:
                    break
                split_pos = match.end()
                sentence = buffer[:split_pos].strip()
                if sentence:
                    # Add emotion prefix for TTS
                    emotion_text = f"{current_emotion}: {sentence}"
                    callback(sentence)
                    tts_queue.put(emotion_text)
                buffer = buffer[split_pos:]
    
    # Process remaining buffer
    if buffer.strip():
        emotion_text = f"{current_emotion}: {buffer.strip()}"
        callback(buffer.strip())
        tts_queue.put(emotion_text)