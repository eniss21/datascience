from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.chatbot.response_generator import Chatbot
from config import (
    INTENTS_PATH, TFIDF_MODEL_PATH, SEMANTIC_MODEL_PATH, RNN_MODEL_PATH,
    API_HOST, API_PORT, MIN_CONFIDENCE_THRESHOLD
)


app = FastAPI(
    title="E-commerce Chatbot API",
    description="NLP-powered customer support chatbot for e-commerce",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
BASE_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Store all chatbots - one per model type
chatbots: Dict[str, Chatbot] = {}

MODEL_PATHS = {
    "tfidf": TFIDF_MODEL_PATH,
    "semantic": SEMANTIC_MODEL_PATH,
    "rnn": RNN_MODEL_PATH
}


class ChatRequest(BaseModel):
    message: str
    model_type: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Where is my order?",
                "model_type": "semantic"
            }
        }


class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    is_fallback: bool
    model_type: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]


class IntentScore(BaseModel):
    intent: str
    score: float


class DebugResponse(BaseModel):
    message: str
    model_type: str
    top_intents: List[IntentScore]
    selected_intent: str
    confidence: float
    response: str


@app.on_event("startup")
async def startup_event():
    """Initialize all chatbot models on startup."""
    global chatbots

    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", MIN_CONFIDENCE_THRESHOLD))

    print("Loading all chatbot models...")

    for model_type, model_path in MODEL_PATHS.items():
        if not Path(model_path).exists():
            print(f"  Skipping {model_type}: model file not found at {model_path}")
            continue

        try:
            print(f"  Loading {model_type} model...")
            chatbots[model_type] = Chatbot(
                intents_path=str(INTENTS_PATH),
                model_path=str(model_path),
                model_type=model_type,
                confidence_threshold=confidence_threshold
            )
            print(f"  {model_type} model loaded successfully")
        except Exception as e:
            print(f"  Failed to load {model_type}: {e}")

    print(f"Chatbot ready! Loaded models: {list(chatbots.keys())}")


def get_chatbot(model_type: Optional[str] = None) -> tuple[Chatbot, str]:
    """Get the appropriate chatbot for the requested model type."""
    if not chatbots:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Default to first available model if not specified
    if model_type is None:
        model_type = list(chatbots.keys())[0]

    model_type = model_type.lower()

    if model_type not in chatbots:
        available = list(chatbots.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_type}' not available. Available models: {available}"
        )

    return chatbots[model_type], model_type


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - serves the web interface."""
    static_file = STATIC_DIR / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "message": "E-commerce Chatbot API",
        "available_models": list(chatbots.keys()),
        "docs": "/docs",
        "health": "/health",
        "web_interface": "/static/index.html" if STATIC_DIR.exists() else None
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={model: True for model in chatbots.keys()}
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Process user message and return chatbot response."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    chatbot, model_type = get_chatbot(request.model_type)
    result = chatbot.chat(request.message.strip())

    return ChatResponse(
        intent=result["intent"],
        confidence=result["confidence"],
        response=result["response"],
        is_fallback=result["is_fallback"],
        model_type=model_type
    )


@app.post("/chat/debug", response_model=DebugResponse, tags=["Chat"])
async def chat_debug(request: ChatRequest):
    """Debug endpoint showing classification details."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    chatbot, model_type = get_chatbot(request.model_type)
    message = request.message.strip()
    top_intents = chatbot.intent_handler.get_top_intents(message, top_k=5)
    result = chatbot.chat(message)

    return DebugResponse(
        message=message,
        model_type=model_type,
        top_intents=[IntentScore(intent=intent, score=score) for intent, score in top_intents],
        selected_intent=result["intent"],
        confidence=result["confidence"],
        response=result["response"]
    )


@app.get("/intents", tags=["Info"])
async def list_intents():
    """List all available intents."""
    if not chatbots:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Use first available chatbot to get intents
    chatbot = list(chatbots.values())[0]
    return {
        "intents": list(chatbot.response_generator.responses.keys())
    }


@app.get("/models", tags=["Info"])
async def list_models():
    """List all available models."""
    return {
        "available_models": list(chatbots.keys()),
        "total": len(chatbots)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
