from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voice_agent.config import settings


class VoiceService(ABC):
    @abstractmethod
    async def text_to_speech(
        self,
        text: str,
        voice_id: str = settings.ELEVENLABS_VOICE_ID,
        model_id: str = settings.ELEVENLABS_TTS_MODEL_ID,
        output_format: str = "mp3_44100_128",
    ) -> AsyncIterator[bytes]:
        ...

    @abstractmethod
    async def speech_to_text(
        self,
        audio_data: bytes,
        audio_filename: str,
        model_id: str = settings.ELEVENLABS_STT_MODEL_ID,
        language_code: str | None = None,
    ) -> dict:
        ...
