import streamlit as st
from audiorecorder import audiorecorder
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import time
import queue
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Initialiser l'état de la session
if "transcription_text" not in st.session_state:
    st.session_state["transcription_text"] = ""

if "recording" not in st.session_state:
    st.session_state["recording"] = False

if "stop_event" not in st.session_state:
    st.session_state["stop_event"] = None

if "audio_queue" not in st.session_state:
    st.session_state["audio_queue"] = queue.Queue()

if "transcription_queue" not in st.session_state:
    st.session_state["transcription_queue"] = queue.Queue()

# Charger le modèle de transcription
@st.cache_resource
def load_model():
    model_id = "bofenghuang/whisper-large-v3-french"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )
    return pipe

model = load_model()

st.title("Transcription en temps réel avec Whisper Large-V3")

# Enregistrement audio via audiorecorder
audio = audiorecorder("Click to record", "Click to stop recording")

# Fonction de transcription
def transcribe_audio(audio_data, transcription_queue):
    audio_segment = AudioSegment.from_wav(audio_data)
    chunks = split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-40)
    
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
        
        result = model(chunk_filename)
        transcription = result["text"]
        transcription_queue.put(transcription)

# Démarrer l'enregistrement
if st.button("Démarrer l'enregistrement") and not st.session_state["recording"]:
    st.session_state["recording"] = True
    st.session_state["transcription_text"] = ""
    st.session_state["stop_event"] = threading.Event()
    
    # Lancer un thread pour la transcription
    threading.Thread(target=transcribe_audio, args=(audio.export(), st.session_state["transcription_queue"]), daemon=True).start()

# Arrêter l'enregistrement
if st.button("Arrêter l'enregistrement") and st.session_state["recording"]:
    st.session_state["stop_event"].set()
    st.session_state["recording"] = False

placeholder = st.empty()

# Afficher la transcription en temps réel
if st.session_state["recording"]:
    while st.session_state["recording"]:
        while not st.session_state["transcription_queue"].empty():
            transcription = st.session_state["transcription_queue"].get()
            st.session_state["transcription_text"] += transcription + " "
        
        # Mise à jour de la transcription en temps réel
        key = f"real_time_transcription_{st.session_state['transcription_key_counter']}"
        st.session_state["transcription_key_counter"] += 1
        placeholder.text_area("Texte transcrit (en temps réel)", st.session_state["transcription_text"], height=200, disabled=True, key=key)
        time.sleep(0.5)

# Affichage de la transcription finale
placeholder.text_area("Texte transcrit", st.session_state["transcription_text"], height=200, disabled=True, key="final_transcription")
