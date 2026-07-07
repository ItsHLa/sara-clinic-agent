import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from voice_agent.agent.chat_agent import ChatAgent
from voice_agent.config import settings
from voice_agent.routers import pipeline, stt, tts
from voice_agent.services.elevenlabs import ElevenLabsService
from voice_agent.services.speechmatics import SpeechmaticsService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("voice_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[APP] Starting up...")
    logger.info("[APP] ELEVENLABS_API_KEY set=%s", bool(settings.ELEVENLABS_API_KEY))
    logger.info("[APP] SPEECHMATICS_API_KEY set=%s", bool(settings.SPEECHMATICS_API_KEY))
    logger.info("[APP] VOICE_PROVIDER set=%s", settings.VOICE_PROVIDER)
    logger.info("[APP] AGENT_URL set=%s", bool(settings.AGENT_URL))
    logger.info("[APP] AGENT_INTERNAL_TOKEN set=%s", bool(settings.AGENT_INTERNAL_TOKEN))
    app.state.voice_services = {
        "elevenlabs": ElevenLabsService(),
        "speechmatics": SpeechmaticsService()
    }

    if settings.VOICE_PROVIDER.lower() == "speechmatics":
        app.state.voice_service = app.state.voice_services["speechmatics"]
    else:
        app.state.voice_service = app.state.voice_services["elevenlabs"]

    app.state.agent_service = ChatAgent()
    logger.info("[APP] Services initialized")
    yield
    logger.info("[APP] Shutting down...")


app = FastAPI(title="Voice Agent API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info("[HTTP] >>> %s %s | Content-Type: %s | Content-Length: %s",
                request.method, request.url.path,
                request.headers.get("content-type", ""),
                request.headers.get("content-length", ""))
    try:
        response = await call_next(request)
        logger.info("[HTTP] <<< %s %s -> %s", request.method, request.url.path, response.status_code)
    except Exception as e:
        logger.error("[HTTP] ERROR %s %s -> %s", request.method, request.url.path, e)
        raise
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(pipeline.router)

logger.info("[APP] Routes registered: POST /agent/chat, /agent/pipeline, /agent/full-cycle, /agent/voice-cycle")


@app.get("/", include_in_schema=False)
async def index():
    logger.info("[APP] Serving index.html")
    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    logger.info("[APP] Starting uvicorn on %s:%s", settings.HOST, settings.PORT)
    uvicorn.run(
        "voice_agent.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
