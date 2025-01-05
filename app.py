import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import tempfile
import threading
import wave
import queue
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence

if "transcription_text" not in st.session_state:
    st.session_state["transcription_text"] = ""

if "recording" not in st.session_state:
    st.session_state["recording"] = False

if "stop_event" not in st.session_state:
    st.session_state["stop_event"] = threading.Event()

if "audio_queue" not in st.session_state:
    st.session_state["audio_queue"] = queue.Queue()

if "transcription_queue" not in st.session_state:
    st.session_state["transcription_queue"] = queue.Queue()

@st.cache_resource
def load_model():
    model_id = "bofenghuang/whisper-large-v3-french"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

model = load_model()

st.title("Transcription en temps réel avec Whisper Large-V3")

sample_rate = 16000
chunk_duration = 4  # Durée des chunks en secondes

def record_audio(queue, stop_event):
    while not stop_event.is_set():
        recording = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            with wave.open(temp_audio.name, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(recording.tobytes())
            queue.put(temp_audio.name)

def transcribe_audio(audio_queue, stop_event, transcription_queue):
    while not stop_event.is_set():
        if not audio_queue.empty():
            audio_file = audio_queue.get()

            audio = AudioSegment.from_wav(audio_file)
            chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=-40)

            for i, chunk in enumerate(chunks):
                chunk_filename = f"chunk_{i}.wav"
                chunk.export(chunk_filename, format="wav")
                
                result = model(chunk_filename)
                transcription = result["text"]
                transcription_queue.put(transcription)

if st.button("Démarrer l'enregistrement") and not st.session_state["recording"]:
    st.session_state["stop_event"].clear()
    st.session_state["recording"] = True
    st.session_state["transcription_text"] = "" 

    threading.Thread(
        target=record_audio,
        args=(st.session_state["audio_queue"], st.session_state["stop_event"]),
        daemon=True
    ).start()
    threading.Thread(
        target=transcribe_audio,
        args=(st.session_state["audio_queue"], st.session_state["stop_event"], st.session_state["transcription_queue"]),
        daemon=True
    ).start()

if st.button("Arrêter l'enregistrement") and st.session_state["recording"]:
    st.session_state["stop_event"].set()
    st.session_state["recording"] = False

placeholder = st.empty()

if "transcription_key_counter" not in st.session_state:
    st.session_state["transcription_key_counter"] = 0

if st.session_state["recording"]:
    while st.session_state["recording"]:
        while not st.session_state["transcription_queue"].empty():
            transcription = st.session_state["transcription_queue"].get()
            st.session_state["transcription_text"] += transcription + " "

        key = f"real_time_transcription_{st.session_state['transcription_key_counter']}"
        st.session_state["transcription_key_counter"] += 1

        placeholder.text_area(
            "Texte transcrit (en temps réel)",
            st.session_state["transcription_text"],
            height=200,
            disabled=True,
            key=key,  
        )
        time.sleep(0.5)

placeholder.text_area(
    "Texte transcrit",
    st.session_state["transcription_text"],
    height=200,
    disabled=True,
    key="final_transcription",  
)
