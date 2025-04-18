# sentence streaming added

import os
import time
import threading
import tempfile
import wave
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import openai
from scipy.io.wavfile import write
from dotenv import load_dotenv
import asyncio
from queue import Queue
import edge_tts
import pyttsx3
import re

# Import our streaming TTS runner.
from junk.streaming_voice import stream_gpt4_response
from junk.transcriber import RealTimeTranscriber

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

# Global recording state and buffer.
recording_active = False
audio_buffer = []  # List of NumPy arrays (each block)
latest_amplitude = 0.0  # Used for waveform display

# Lock for thread-safe updates.
import threading
buffer_lock = threading.Lock()


# Create global transcriber instance
transcriber = RealTimeTranscriber()

# sounddevice callback: append incoming audio block and update amplitude.
def audio_callback(indata, frames, time_info, status):
    global audio_buffer, latest_amplitude
    if recording_active:
        with buffer_lock:
            audio_buffer.append(indata.copy())
        latest_amplitude = np.abs(indata).mean()

        # Stream to transcriber (chunk by chunk)
        transcriber.add_audio_chunk(indata, SAMPLE_RATE)

# Function to start recording.
def start_recording():
    global recording_active, audio_buffer
    recording_active = True
    with buffer_lock:
        audio_buffer = []  # clear previous recording
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                            blocksize=BLOCK_SIZE, callback=audio_callback)
    stream.start()
    return stream

# Function to stop recording.
def stop_recording(stream):
    global recording_active
    recording_active = False
    stream.stop()
    stream.close()

# Save accumulated audio to a temporary WAV file.
def save_audio_to_wav():
    with buffer_lock:
        if not audio_buffer:
            return None
        data = np.concatenate(audio_buffer, axis=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav_path = f.name
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes per sample.
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())
    return wav_path

# Transcribe audio using OpenAI Whisper.
def transcribe_audio(wav_path):
    with open(wav_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"].strip()

# Tkinter-based GUI application.
class VoiceAssistantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Assistant")
        self.geometry("500x350")
        self.resizable(False, False)
        
        self.status_var = tk.StringVar(value="Click 'Start Recording' to begin.")
        self.status_label = ttk.Label(self, textvariable=self.status_var, font=("Helvetica", 12))
        self.status_label.pack(pady=10)
        
        # Canvas for waveform indicator.
        self.canvas = tk.Canvas(self, width=400, height=100, bg="black")
        self.canvas.pack(pady=10)
        
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        self.start_button = ttk.Button(button_frame, text="Start Recording", command=self.start_recording_handler)
        self.start_button.grid(row=0, column=0, padx=10)
        self.stop_button = ttk.Button(button_frame, text="Stop Recording & Submit", command=self.stop_and_process_handler)
        self.stop_button.grid(row=0, column=1, padx=10)
        self.exit_button = ttk.Button(self, text="Exit", command=self.destroy)
        self.exit_button.pack(pady=10)
        
        self.rec_stream = None
        self.update_waveform()
        
    def update_waveform(self):
        self.canvas.delete("all")
        # Scale the amplitude (max for 16-bit is ~32767) to canvas width 400.
        bar_width = int((latest_amplitude / 32767) * 400)
        self.canvas.create_rectangle(0, 0, bar_width, 100, fill="green")
        self.after(50, self.update_waveform)
        
    def start_recording_handler(self):
        global recording_active
        if not recording_active:
            self.status_var.set("Recording... Please speak your command.")
            self.rec_stream = start_recording()
    
    def stop_and_process_handler(self):
        if recording_active and self.rec_stream is not None:
            stop_recording(self.rec_stream)
            self.status_var.set("Recording stopped. Processing audio...")
            threading.Thread(target=self.process_recording, daemon=True).start()
        else:
            self.status_var.set("No recording in progress.")
    def process_recording(self):
        # Get the transcription already streamed during recording
        transcribed_text = transcriber.get_current_transcription()
        if not transcribed_text:
            self.status_var.set("No speech detected. Please try again.")
            return
        
        transcriber.reset()  # Clear for next session
        self.status_var.set("Generating response...")

        def on_sentence(sentence):
            self.status_var.set(f"Speaking: {sentence}")
        
        try:
            threading.Thread(
                target=stream_gpt4_response,
                args=(transcribed_text, on_sentence),
                daemon=True
            ).start()
        except Exception as e:
            self.status_var.set(f"Error: {e}")

        
        # os.remove(wav_path)
        
if __name__ == "__main__":
    app = VoiceAssistantApp()
    def on_closing():
        transcriber.stop()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()

    # app.mainloop()
