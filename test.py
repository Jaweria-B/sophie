import os
import time
import threading
import tempfile
import wave
import queue
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import openai
import pyttsx3
from faster_whisper import WhisperModel  # pip install faster-whisper
from dotenv import load_dotenv

# ─── Environment & Keys ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# ─── Recording Config ──────────────────────────────────────────────────────────
SAMPLE_RATE = 44100  # Hz
CHANNELS    = 1
DTYPE       = 'int16'
BLOCK_SIZE  = 1024  # frames per callback

# ─── Globals & Thread Safety ────────────────────────────────────────────────────
recording_active = False
audio_buffer     = []            # list of NumPy arrays
buffer_lock      = threading.Lock()
transcript_queue = queue.Queue() # holds new partial transcripts

# ─── Load Faster‑Whisper Model (GPU if available) ───────────────────────────────
device      = "cuda" if sd.default.device[0] is not None and sd.default.device else "cpu"
model_size  = "medium.en" if device=="cuda" else "small.en"
compute_type= "float16"      if device=="cuda" else "int8"
whisper_model = WhisperModel(
    model_size,
    device=device,
    compute_type=compute_type
)

# ─── TTS Engine (pyttsx3) ───────────────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
tts_engine.setProperty("volume", 1.0)

def speak(text: str):
    """Speak text in background thread."""
    threading.Thread(target=lambda: (tts_engine.say(text), tts_engine.runAndWait()), daemon=True).start()

# ─── Audio Callback ─────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    """sounddevice callback: append incoming audio to buffer."""
    global audio_buffer
    if recording_active:
        with buffer_lock:
            audio_buffer.append(indata.copy())

# ─── Chunking Worker ───────────────────────────────────────────────────────────
def chunking_worker():
    """
    While recording, every CHUNK_DURATION seconds dump buffered audio
    to a WAV and submit to transcription_worker via a file queue.
    """
    CHUNK_DURATION = 1.0  # seconds
    while recording_active:
        time.sleep(CHUNK_DURATION)
        with buffer_lock:
            if not audio_buffer:
                continue
            data = np.concatenate(audio_buffer, axis=0)
            audio_buffer.clear()
        # Write to temp WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
            wf = wave.open(path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16‑bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(data.tobytes())
            wf.close()
        # Enqueue for transcription
        transcript_queue.put(path)

# ─── Transcription Worker ─────────────────────────────────────────────────────
accumulated_text = ""  # holds the live transcript

def transcription_worker():
    """
    Pull WAV paths from transcript_queue, transcribe via Faster‑Whisper
    streaming, and update accumulated_text and GUI via a callback.
    """
    global accumulated_text
    while recording_active or not transcript_queue.empty():
        try:
            wav_path = transcript_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Transcribe small audio file
        segments, _ = whisper_model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True,
            language="en"
        )
        os.remove(wav_path)

        # Append and push partial text
        partial = "".join(segment.text for segment in segments).strip()
        if partial:
            accumulated_text += " " + partial
            gui_app.queue_transcript_update(accumulated_text.strip())

# ─── Tkinter GUI App ───────────────────────────────────────────────────────────
class VoiceAssistantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real‑Time Voice Assistant")
        self.geometry("600x400")
        self.resizable(False, False)

        self.status_var   = tk.StringVar("Press ‘Start Recording’")
        self.transcript_var = tk.StringVar("")

        ttk.Label(self, textvariable=self.status_var).pack(pady=5)
        ttk.Label(self, text="Live Transcript:", font=("Arial", 10, "bold")).pack()
        ttk.Label(self, textvariable=self.transcript_var, wraplength=580).pack(pady=5)

        # Waveform Canvas
        self.canvas = tk.Canvas(self, width=580, height=80, bg="black")
        self.canvas.pack(pady=5)

        # Buttons
        frm = ttk.Frame(self)
        frm.pack(pady=10)
        ttk.Button(frm, text="Start Recording", command=self.start_recording).grid(row=0, column=0, padx=5)
        ttk.Button(frm, text="Stop & Submit",    command=self.stop_and_submit).grid(row=0, column=1, padx=5)
        ttk.Button(frm, text="Exit",             command=self.destroy).grid(row=0, column=2, padx=5)

        self.update_waveform()
        self.poll_transcript()

    def update_waveform(self):
        """Draw simple amplitude bar."""
        with buffer_lock:
            amp = np.abs(audio_buffer[-1]).mean() if audio_buffer else 0
        width = int((amp / 32767) * 580)
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, 80, fill="green")
        self.after(50, self.update_waveform)

    def queue_transcript_update(self, text):
        """Called from transcription thread to update GUI."""
        self.transcript_var.set(text)

    def poll_transcript(self):
        """Periodically update GUI from queue if needed."""
        # (We use direct setter in this design.)
        self.after(100, self.poll_transcript)

    def start_recording(self):
        global recording_active, audio_buffer
        if not recording_active:
            recording_active = True
            audio_buffer = []

            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype=DTYPE, blocksize=BLOCK_SIZE,
                callback=audio_callback
            )
            self.stream.start()

            # Start background workers
            threading.Thread(target=chunking_worker,    daemon=True).start()
            threading.Thread(target=transcription_worker,daemon=True).start()

            self.status_var.set("Recording… Speak now!")

    def stop_and_submit(self):
        global recording_active
        if recording_active:
            recording_active = False
            self.stream.stop()
            self.stream.close()
            self.status_var.set("Recording stopped. Sending to GPT‑4…")

            # Final submission of accumulated_text
            prompt = self.transcript_var.get()
            threading.Thread(target=self._send_to_gpt4, args=(prompt,), daemon=True).start()
        else:
            self.status_var.set("Not currently recording.")

    def _send_to_gpt4(self, prompt):
        """Send final transcript to GPT‑4 and speak response."""
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user","content":prompt}],
                stream=False
            )
            reply = resp.choices[0].message.content.strip()
            self.status_var.set("GPT‑4 responded. Speaking…")
            speak(reply)
            self.status_var.set("Done.")
        except Exception as e:
            self.status_var.set(f"Error: {e}")

# ─── Launch ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui_app = VoiceAssistantApp()
    gui_app.mainloop()
