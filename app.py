import os
import openai
import pvporcupine
import pyaudio
import wave
import numpy as np  # Import numpy for conversion
from src.agents.agent import Agent
from src.prompts.prompts import assistant_prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key for chat and Whisper
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the agent with the OpenAI model (e.g., "gpt-4")
agent = Agent(name="OpenAI-Agent", model="gpt-4", system_prompt=assistant_prompt)

# Set wake word configuration (using Porcupine)
# For a custom wake word like "Hi Sophie", you may need a custom model.
# Here we use a placeholder keyword ("hey siri") with the required access key.
porcupine = pvporcupine.create(
    keywords=["hey siri"],
    access_key=os.getenv("PORCUPINE_API_KEY")
)

# Initialize PyAudio with the sample rate and frame length from Porcupine
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Voice Assistant is listening for the wake word...")

def listen_for_wake_word():
    """
    Continuously listens for the wake word.
    When detected, returns control.
    """
    while True:
        # Read raw audio data
        pcm_data = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        # Convert the raw bytes into an array of int16 samples
        pcm = np.frombuffer(pcm_data, dtype=np.int16)
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("\nWake word detected! Please speak your command.")
            return

def record_command(duration=5, filename="command.wav"):
    """
    Records audio from the microphone for a fixed duration and saves it as a WAV file.
    The recording is configured at 16kHz mono (compatible with OpenAI Whisper).
    Uses porcupine.frame_length as the chunk size.
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Whisper works best with 16kHz audio
    RECORD_SECONDS = duration
    frames = []
    
    # Use the same chunk size as Porcupine expects
    CHUNK = porcupine.frame_length  
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("Recording command...")
    num_frames = int(RATE / CHUNK * RECORD_SECONDS)
    for _ in range(num_frames):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    
    # Save the recorded audio to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

def transcribe_command(audio_filename):
    """
    Uses OpenAI Whisper to transcribe the recorded audio command.
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
