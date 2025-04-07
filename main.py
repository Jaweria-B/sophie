import os
import openai
import pvporcupine
import pyaudio
import struct
import wave
from src.agents.agent import Agent
from src.prompts.prompts import assistant_prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key for both chat and Whisper
openai.api_key = os.getenv("OPENAI_API_KEY")

# IMPORTANT: Pin openai to version 0.28.0 to use the legacy Audio API.
# Run: pip install openai==0.28.0

# Initialize the agent with the OpenAI model (e.g., "gpt-4")
agent = Agent(name="OpenAI-Agent", model="gpt-4", system_prompt=assistant_prompt)

# Set wake word configuration
WAKE_WORD = "Hi Sophie"
# For this example, we use a built-in keyword. Adjust the keyword and access_key as needed.
porcupine = pvporcupine.create(keywords=["hey siri"], access_key=os.getenv("PORCUPINE_API_KEY"))

pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length  # Should match Porcupine's expected frame length
)

print("Voice Assistant is listening for the wake word...")

def listen_for_wake_word():
    while True:
        pcm_bytes = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        # Convert byte data to 16-bit integers (expected by Porcupine)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_bytes)
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("\nWake word detected! Say your command:")
            return

def record_command(duration=5, filename="command.wav"):
    """
    Records audio from the microphone for a fixed duration and saves it as a WAV file.
    The recording is configured at 16kHz mono, which works well with OpenAI Whisper.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = duration
    frames = []
    
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)
    print("Recording command...")
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def transcribe_command(audio_filename):
    """
    Uses OpenAI Whisper (legacy API) to transcribe the recorded audio command.
    Note: This function requires openai==0.28.0.
    """
    with open(audio_filename, "rb") as audio_file:
        print("Transcribing command using OpenAI Whisper...")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def main():
    while True:
        # Wait for the wake word to be detected
        listen_for_wake_word()
        
        # Record the spoken command after the wake word is detected
        audio_file = record_command(duration=5, filename="command.wav")
        
        # Transcribe the recorded audio to text using Whisper
        command_text = transcribe_command(audio_file)
        print("Transcribed Command:", command_text)
        
        if command_text.lower() in ["goodbye", "exit", "quit"]:
            print("Assistant: Goodbye!")
            break
        
        # Invoke the agent with the transcribed command
        response = agent.invoke(command_text)
        print("Assistant:", response)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        porcupine.delete()
