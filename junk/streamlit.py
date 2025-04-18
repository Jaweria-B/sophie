import os
import time
import wave
import tempfile
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import openai
import pyttsx3
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Load environment variables and OpenAI API key.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.title("Real-Time Voice Assistant")
st.markdown("""
**Instructions:**

1. Click **Start Recording** to begin capturing your voice.
2. When you finish speaking, click **Stop Recording**.
3. The app will transcribe your speech using Whisper, send your command to GPT‑4, and deliver a voice response.
""")

# ---------------------------
# Session State Initialization
# ---------------------------
if "recording" not in st.session_state:
    st.session_state.recording = False

# ---------------------------
# Custom Audio Processor: Accumulate Audio Frames
# ---------------------------
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []      # List to store captured audio frames
        self.record = False   # Flag to control recording

    def recv_audio(self, frame):
        # Convert incoming frame to a NumPy array.
        arr = frame.to_ndarray()
        if self.record:
            self.frames.append(arr)
        return frame

# ---------------------------
# Start the WebRTC Streamer
# ---------------------------
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": {"sampleRate": 16000, "channelCount": 1}, "video": False},
    async_processing=True,
)

# ---------------------------
# Recording Control Buttons
# ---------------------------
col_start, col_stop = st.columns(2)
if col_start.button("Start Recording"):
    if webrtc_ctx.audio_processor:
        # Set recording flag and clear previous frames.
        webrtc_ctx.audio_processor.record = True
        webrtc_ctx.audio_processor.frames = []
        st.session_state.recording = True
        st.success("Recording started.")
if col_stop.button("Stop Recording"):
    if webrtc_ctx.audio_processor:
        webrtc_ctx.audio_processor.record = False
        st.session_state.recording = False
        st.success("Recording stopped.")

# ---------------------------
# Process Recorded Audio When Recording is Stopped
# ---------------------------
if not st.session_state.recording and webrtc_ctx.audio_processor:
    frames = webrtc_ctx.audio_processor.frames
    if frames:
        st.info("Processing recorded audio...")
        try:
            # Concatenate the recorded frames along the time axis.
            audio_data = np.concatenate(frames, axis=0)
        except Exception as e:
            st.error(f"Error combining audio frames: {e}")
            audio_data = None

        if audio_data is not None:
            # Save the audio data to a temporary WAV file.
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                    wav_path = wav_file.name
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)       # Mono audio
                    wf.setsampwidth(2)         # 16-bit PCM = 2 bytes per sample
                    wf.setframerate(16000)     # 16 kHz sample rate
                    wf.writeframes(audio_data.tobytes())
                st.success("Audio file saved successfully.")
            except Exception as e:
                st.error(f"Error writing WAV file: {e}")
                wav_path = None

            if wav_path:
                # ---------------------------
                # Transcription using OpenAI Whisper
                # ---------------------------
                st.info("Transcribing audio...")
                try:
                    with open(wav_path, "rb") as audio_file:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    command_text = transcript["text"].strip()
                    st.write("**Transcribed Command:**", command_text)
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    command_text = ""

                # ---------------------------
                # Processing Command via GPT-4
                # ---------------------------
                if command_text:
                    st.info("Generating response from GPT-4...")
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": command_text}],
                            max_tokens=150
                        )
                        response_text = response.choices[0].message.content.strip()
                        st.write("**Response:**", response_text)
                    except Exception as e:
                        st.error(f"GPT-4 error: {e}")
                        response_text = "I'm sorry, I couldn't process your command."
                else:
                    response_text = "No valid command recognized."
                
                # ---------------------------
                # Text-to-Speech (TTS) via pyttsx3
                # ---------------------------
                st.info("Generating voice response...")
                try:
                    tts_engine = pyttsx3.init()
                    tts_engine.setProperty("rate", 150)
                    tts_engine.setProperty("volume", 1.0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                        tts_path = tts_file.name
                    tts_engine.save_to_file(response_text, tts_path)
                    tts_engine.runAndWait()
                    with open(tts_path, "rb") as f:
                        tts_audio_bytes = f.read()
                    st.audio(tts_audio_bytes, format="audio/mp3")
                    os.remove(tts_path)
                except Exception as e:
                    st.error(f"TTS generation error: {e}")
                
                # Clean up temporary WAV file.
                os.remove(wav_path)
                
                # Reset frames so processing doesn't repeat.
                webrtc_ctx.audio_processor.frames = []
            else:
                st.error("WAV file creation failed.")
    else:
        st.info("No audio recorded yet.")
