from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voice_agent.config import settings


class VoiceService(ABC):
    @abstractmethod
    async def text_to_speech(
        self,
        text,
        voice_id=settings.ELEVENLABS_VOICE_ID,
        model_id=settings.ELEVENLABS_TTS_MODEL_ID,
        output_format="mp3_44100_128",
    ):
        ...

    @abstractmethod
    async def speech_to_text(
        self,
        audio_data,
        audio_filename,
        model_id=settings.ELEVENLABS_STT_MODEL_ID,
        language_code=None,
    ):
        ...
