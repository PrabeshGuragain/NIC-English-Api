from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchaudio
import numpy as np
from pathlib import Path
import aiofiles
from typing import Optional
import eng_to_ipa as ipa
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech to Text API")

# Initialize paths
MODELS_DIR = Path("silero-models")
TEMP_DIR = Path("temp")
SAMPLE_RATE = 16000

# Create necessary directories
TEMP_DIR.mkdir(exist_ok=True)

class TranscriptionResponse(BaseModel):
    text: str
    success: bool
    phonetic: Optional[str] = None
    error: Optional[str] = None

def check_dependencies():
    """Check and verify all required dependencies"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info("CUDA is available")
        else:
            logger.info("Running on CPU")
            
        try:
            import soundfile
            logger.info("soundfile backend available")
        except ImportError:
            logger.warning("soundfile not available, installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
            
    except Exception as e:
        logger.error(f"Dependency check failed: {str(e)}")
        raise

async def initialize_model():
    """Initialize the Silero STT model"""
    try:
        if not MODELS_DIR.exists():
            logger.info("Downloading Silero models...")
            os.system("git clone -q --depth 1 https://github.com/snakers4/silero-models")
        
        # Add silero-models to path for imports
        sys.path.append(str(MODELS_DIR))
        
        global model, decoder, utils
        
        # Load model configuration
        logger.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model directly using torch hub
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                             model='silero_stt',
                                             language='en',
                                             device=device)
        
        # Define transcription function
        def transcribe_audio(audio_path: str) -> str:
            try:
                # Read audio file
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if sample_rate != SAMPLE_RATE:
                    logger.info(f"Resampling from {sample_rate} to {SAMPLE_RATE}")
                    resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    logger.info("Converting stereo to mono")
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Move to device and ensure correct shape
                waveform = waveform.to(device)
                
                # Transcribe
                logger.info("Transcribing audio...")
                emission = model(waveform)
                transcription = decoder(emission[0])  # Decode the logits into text
                logger.info("Transcription complete")
                
                return transcription
                
            except Exception as e:
                logger.error(f"Audio transcription failed: {str(e)}")
                raise
        
        global transcribe
        transcribe = transcribe_audio
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    try:
        logger.info("Starting up the application...")
        check_dependencies()
        await initialize_model()
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary directory"""
    try:
        temp_file = TEMP_DIR / upload_file.filename
        async with aiofiles.open(temp_file, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
        return temp_file
    except Exception as e:
        logger.error(f"File save failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

@app.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file
    """
    try:
        if not file.filename.endswith(('.wav', '.mp3')):
            raise HTTPException(
                status_code=400,
                detail="Only .wav and .mp3 files are supported"
            )

        # Save uploaded file
        temp_file = await save_upload_file(file)
        
        try:
            # Transcribe audio
            transcription = transcribe(str(temp_file))
            logger.info(f"Transcription result: {transcription}")
            
            return TranscriptionResponse(
                text=transcription,
                phonetic=ipa.convert(transcription),
                success=True
            )
            
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Error processing audio: {str(e)}"
            }
        )

@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {
        "status": "healthy",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }