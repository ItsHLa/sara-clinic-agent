from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from voice_agent.schemas.voice import STTResponse, WordTiming
from voice_agent.deps import get_voice_service
from voice_agent.services.base import VoiceService

router = APIRouter(prefix="/stt", tags=["stt"])


@router.post("/transcribe", response_model=STTResponse)
async def transcribe(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    service: Annotated[VoiceService, Depends(get_voice_service)],
    model_id: str = Form("scribe_v2"),
    language_code: str | None = Form(None),
):
    audio_data = await file.read()
    result = await service.speech_to_text(
        audio_data=audio_data,
        audio_filename=file.filename or "audio.wav",
        model_id=model_id,
        language_code=language_code,
    )

    words = None
    if "words" in result:
        words = [WordTiming(**w) for w in result["words"]]

    return STTResponse(
        text=result["text"],
        language_code=result.get("language_code"),
        language_probability=result.get("language_probability"),
        words=words,
    )
