import os
import time
import threading
import tempfile
import wave
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
import openai
import pyttsx3
from scipy.io.wavfile import write
from dotenv import load_dotenv

# Load environment variables and OpenAI API key.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise Exception("OPENAI_API_KEY not set in .env file.")

# Configuration for recording.
SAMPLE_RATE = 44100  # Hz
CHANNELS = 1
DTYPE = 'int16'
BLOCK_SIZE = 1024  # samples per block

# Global variables to hold recording state.
recording_active = False
audio_buffer = []   # list of NumPy arrays (each block)
latest_amplitude = 0.0  # used for waveform display

# Lock for thread-safe updates.
buffer_lock = threading.Lock()

# Initialize pyttsx3 TTS engine (will be used later in a worker thread).
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
tts_engine.setProperty("volume", 1.0)

# Sounddevice callback: append each incoming audio block and update amplitude.
def audio_callback(indata, frames, time_info, status):
    global audio_buffer, latest_amplitude
    if recording_active:
        with buffer_lock:
            audio_buffer.append(indata.copy())
        # Update latest amplitude as the mean absolute value.
        latest_amplitude = np.abs(indata).mean()

# Function to start recording.
def start_recording():
    global recording_active, audio_buffer
    recording_active = True
    with buffer_lock:
        audio_buffer = []  # clear previous recording
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype=DTYPE, blocksize=BLOCK_SIZE,
                            callback=audio_callback)
    stream.start()
    return stream

# Function to stop recording.
def stop_recording(stream):
    global recording_active
    recording_active = False
    stream.stop()
    stream.close()

# Save recorded audio to a temporary WAV file.
def save_audio_to_wav():
    with buffer_lock:
        if not audio_buffer:
            return None
        data = np.concatenate(audio_buffer, axis=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav_path = f.name
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM: 2 bytes per sample.
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())
    return wav_path

# Function to transcribe audio using OpenAI Whisper.
def transcribe_audio(wav_path):
    with open(wav_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"].strip()

# Function to call GPT-4 with transcribed text.
def get_gpt4_response(command_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": command_text}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Function to speak response using pyttsx3.
def speak_response(response_text):
    # This function will run in a worker thread so that TTS doesn't block the GUI.
    try:
        tts_engine.say(response_text)
        tts_engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# Tkinter-based GUI application.
class VoiceAssistantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Assistant")
        self.geometry("500x300")
        self.resizable(False, False)
        
        # Status label.
        self.status_var = tk.StringVar(value="Click 'Start Recording' to begin.")
        self.status_label = ttk.Label(self, textvariable=self.status_var, font=("Helvetica", 12))
        self.status_label.pack(pady=10)
        
        # Canvas for waveform indicator.
        self.canvas = tk.Canvas(self, width=400, height=100, bg="black")
        self.canvas.pack(pady=10)
        
        # Buttons.
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        self.start_button = ttk.Button(button_frame, text="Start Recording", command=self.start_recording_handler)
        self.start_button.grid(row=0, column=0, padx=10)
        self.stop_button = ttk.Button(button_frame, text="Stop Recording & Process", command=self.stop_and_process_handler)
        self.stop_button.grid(row=0, column=1, padx=10)
        
        self.exit_button = ttk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=10)
        
        # Recording stream placeholder.
        self.rec_stream = None
        
        # Schedule waveform update.
        self.update_waveform()
        
    def update_waveform(self):
        # Update a simple waveform indicator on the canvas based on latest_amplitude.
        self.canvas.delete("all")
        # Scale amplitude (max for 16-bit is ~32767) to canvas width (400).
        bar_width = int((latest_amplitude / 32767) * 400)
        self.canvas.create_rectangle(0, 0, bar_width, 100, fill="green")
        # Schedule next update.
        self.after(50, self.update_waveform)
        
    def start_recording_handler(self):
        global recording_active
        if not recording_active:
            self.status_var.set("Recording... Please speak your command.")
            self.rec_stream = start_recording()
    
    def stop_and_process_handler(self):
        if recording_active and self.rec_stream is not None:
            stop_recording(self.rec_stream)
            self.status_var.set("Recording stopped. Processing...")
            # Process the recording in a separate thread.
            threading.Thread(target=self.process_recording, daemon=True).start()
        else:
            self.status_var.set("No recording in progress.")
    
    def process_recording(self):
        wav_path = save_audio_to_wav()
        if not wav_path:
            self.status_var.set("No audio recorded. Please try again.")
            return
        
        # Transcribe the audio.
        self.status_var.set("Transcribing audio...")
        try:
            transcribed_text = transcribe_audio(wav_path)
        except Exception as e:
            self.status_var.set(f"Transcription error: {e}")
            os.remove(wav_path)
            return
        
        # Get GPT-4 response.
        self.status_var.set("Getting response from GPT-4...")
        try:
            response_text = get_gpt4_response(transcribed_text)
        except Exception as e:
            self.status_var.set(f"GPT-4 error: {e}")
            os.remove(wav_path)
            return
        
        # Speak the response.
        self.status_var.set("Speaking the response...")
        threading.Thread(target=speak_response, args=(response_text,), daemon=True).start()
        self.status_var.set("Done.")
        # Clean up.
        os.remove(wav_path)

if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.mainloop()
