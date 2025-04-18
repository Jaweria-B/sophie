# real_time_transcriber.py
import os
import threading
import queue
import time
import tempfile
import wave
import numpy as np
from faster_whisper import WhisperModel
import torch

class RealTimeTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = "medium.en" if self.device == "cuda" else "small.en"
        self.model = WhisperModel(self.model_size, device=self.device, compute_type="float16" if self.device == "cuda" else "int8")
        self.audio_queue = queue.Queue()
        self.transcribed_text = ""
        self.running = True
        self.worker_thread = threading.Thread(target=self.transcribe_loop, daemon=True)
        self.worker_thread.start()

    def add_audio_chunk(self, audio_chunk: np.ndarray, samplerate: int):
        self.audio_queue.put((audio_chunk.copy(), samplerate))

    def transcribe_loop(self):
        while self.running:
            try:
                chunk, samplerate = self.audio_queue.get(timeout=0.1)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    self._save_wav(tmpfile.name, chunk, samplerate)
                    segments, _ = self.model.transcribe(tmpfile.name, beam_size=1, vad_filter=True)
                    for seg in segments:
                        self.transcribed_text += seg.text.strip() + " "
                os.remove(tmpfile.name)
            except queue.Empty:
                continue

    def get_current_transcription(self):
        return self.transcribed_text.strip()

    def reset(self):
        self.transcribed_text = ""

    def stop(self):
        self.running = False
        self.worker_thread.join()

    def _save_wav(self, path, data, samplerate):
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(data.tobytes())
