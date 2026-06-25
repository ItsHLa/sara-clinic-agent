import logging
from collections.abc import AsyncIterator

import httpx

from voice_agent.config import settings
from voice_agent.services.base import VoiceService

logger = logging.getLogger("voice_agent.elevenlabs")


class ElevenLabsService(VoiceService):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.base_url = settings.ELEVENLABS_BASE_URL
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info("[ELEVENLABS] Initialized with api_key=%s...", self.api_key[:8] if self.api_key else "NONE")

    async def text_to_speech(
        self,
        text: str,
        voice_id: str = settings.ELEVENLABS_VOICE_ID,
        model_id: str = settings.ELEVENLABS_TTS_MODEL_ID,
        output_format: str = "mp3_44100_128",
    ) -> AsyncIterator[bytes]:
        url = f"{self.base_url}/v1/text-to-speech/{voice_id}/stream"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"text": text, "model_id": model_id}
        params = {"output_format": output_format}

        async with self._client.stream("POST", url, headers=headers, json=payload, params=params) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error("[ELEVENLABS:TTS] Failed: status=%d body=%s", response.status_code, body[:200])
                raise httpx.HTTPStatusError(f"ElevenLabs TTS returned {response.status_code}", request=response.request, response=response)
            async for chunk in response.aiter_bytes():
                yield chunk

    async def speech_to_text(
        self,
        audio_data: bytes,
        audio_filename: str,
        model_id: str = settings.ELEVENLABS_STT_MODEL_ID,
        language_code: str | None = None,
    ) -> dict:
        url = f"{self.base_url}/v1/speech-to-text"

        headers = {"xi-api-key": self.api_key}

        data: dict[str, str | None] = {"model_id": model_id}
        if language_code:
            data["language_code"] = language_code

        files = {"file": (audio_filename, audio_data, "audio/wav")}
        for key, value in data.items():
            files[key] = (None, value)

        response = await self._client.post(url, headers=headers, files=files)
        if response.status_code != 200:
            logger.error("[ELEVENLABS:STT] Failed: status=%d body=%s", response.status_code, response.text[:300])
            raise httpx.HTTPStatusError(
                f"ElevenLabs STT returned {response.status_code}",
                request=response.request, response=response,
            )
        return response.json()
