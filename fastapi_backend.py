"""
fastapi whisper transcriber
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import uvicorn
import io
import logging
import numpy as np
import soundfile as sf  # This will help in reading WAV files
from fastapi.middleware.cors import CORSMiddleware
# Set up basic logging
logging.basicConfig(level=logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Actions to perform at application startup
    """
    print("Starting up...")

    # Switch to a more reliable Whisper model
    model_id = "bofenghuang/whisper-large-v3-french"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        global model_pipeline
        model_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    except Exception as e:
        logging.error(f"Error loading model or processor: {e}")
        yield
        return

    yield  # The application runs while this generator is suspended
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def read_root():
    """
    Welcome message
    """
    return {"message": "Welcome to the Transcription API"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Endpoint to transcribe uploaded audio chunk
    """
    try:
        # Read the uploaded file
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)

        # Log the file size to check if it's received correctly
        logging.debug(f"Received file of size {len(audio_bytes)} bytes")

        # Load the audio data using soundfile (sf)
        audio_data, sample_rate = sf.read(audio_file)

        # Log the audio sample rate and shape
        logging.debug(f"Audio sample rate: {sample_rate}, audio shape: {audio_data.shape}")

        # Preprocess the audio and use the model for transcription
        transcription = model_pipeline(audio_data)["text"]
        logging.debug(f"Transcription: {transcription}")

        # Return the transcription as part of the response
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        logging.error("Error processing transcription: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

def main():
    """Run the FastAPI app using uvicorn."""
    uvicorn.run("fastapi_backend:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
