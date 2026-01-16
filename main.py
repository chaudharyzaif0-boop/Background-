from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bharat STT API",
    description="Speech to Text API for Indian Languages",
    version="1.0.0"
)

# CORS - Mobile app se call karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me specific domain dalo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper model load (server start pe ek baar)
logger.info("Loading Whisper model...")
try:
    model = whisper.load_model("base")  # base = good balance
    logger.info("✅ Whisper model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None

# Supported languages
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "en": "English", 
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi"
}

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "🎤 Bharat STT API",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "transcribe": "/api/stt",
            "languages": "/api/languages"
        }
    }

@app.get("/health")
async def health_check():
    """Server health check"""
    return {
        "ok": True,
        "status": "running",
        "model_loaded": model is not None,
        "model_type": "whisper-base"
    }

@app.get("/api/languages")
async def get_languages():
    """Get supported languages"""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "total": len(SUPPORTED_LANGUAGES)
    }

@app.post("/api/stt")
async def speech_to_text(
    audio: UploadFile = File(..., description="Audio file (m4a, wav, mp3)"),
    language: str = Form("hi", description="Language code (hi, en, ta, etc.)")
):
    """
    Main STT endpoint
    - Upload audio file
    - Get transcribed text
    """
    
    # Model check
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Server starting..."
        )
    
    # Language validation
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{language}' not supported. Use: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    # File format validation
    allowed_formats = ['.m4a', '.wav', '.mp3', '.ogg', '.flac']
    file_ext = os.path.splitext(audio.filename)[1].lower()
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Allowed: {allowed_formats}"
        )
    
    temp_file = None
    
    try:
        # Temporary file banao
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Audio file save karo
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f"Processing audio: {audio.filename} | Language: {language}")
        
        # Whisper se transcribe karo
        result = model.transcribe(
            temp_path,
            language=language,
            fp16=False,  # CPU compatibility
            verbose=False
        )
        
        transcribed_text = result["text"].strip()
        
        logger.info(f"✅ Transcription successful: {transcribed_text[:50]}...")
        
        # Response
        return JSONResponse({
            "success": True,
            "text": transcribed_text,
            "language": language,
            "language_name": SUPPORTED_LANGUAGES[language],
            "filename": audio.filename
        })
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    
    finally:
        # Cleanup - temp file delete karo
        if temp_file and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("🗑️ Temporary file deleted")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")

# Server start
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        reload=True  # Development mode
    )
