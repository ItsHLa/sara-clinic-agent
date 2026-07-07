from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from voice_agent.schemas.voice import TTSRequest, TTSStatus
from voice_agent.deps import get_voice_service
from voice_agent.services.base import VoiceService

router = APIRouter(prefix="/tts", tags=["tts"])


@router.websocket("/stream")
async def tts_stream(
    websocket: WebSocket,
    service: Annotated[VoiceService, Depends(get_voice_service)],
):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        request = TTSRequest(**data)

        status = TTSStatus(status="start", message="Streaming audio")
        await websocket.send_text(status.model_dump_json())

        async for chunk in service.text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id,
            output_format=request.output_format,
        ):
            await websocket.send_bytes(chunk)

        status = TTSStatus(status="end", message="Stream complete")
        await websocket.send_text(status.model_dump_json())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        status = TTSStatus(status="error", message=str(e))
        await websocket.send_text(status.model_dump_json())
