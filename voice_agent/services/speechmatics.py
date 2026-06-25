import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx

from voice_agent.config import settings
from voice_agent.services.base import VoiceService

logger = logging.getLogger("voice_agent.speechmatics")


class SpeechmaticsService(VoiceService):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.SPEECHMATICS_API_KEY
        self.tts_base_url = settings.SPEECHMATICS_TTS_BASE_URL
        self.stt_base_url = settings.SPEECHMATICS_STT_BASE_URL
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info("[SPEECHMATICS] Initialized with api_key=%s...", self.api_key[:8] if self.api_key else "NONE")

    async def text_to_speech(
        self,
        text: str,
        voice_id: str = settings.SPEECHMATICS_TTS_VOICE_ID,
        model_id: str = "",
        output_format: str = "mp3_44100_128",
    ) -> AsyncIterator[bytes]:
        url = f"{self.tts_base_url}/generate/{voice_id}"

        if not self.api_key:
            raise ValueError("SPEECHMATICS_API_KEY is not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        async with self._client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error("[SPEECHMATICS:TTS] Failed: status=%d body=%s", response.status_code, body[:200])
                raise httpx.HTTPStatusError(f"Speechmatics TTS returned {response.status_code}", request=response.request, response=response)
            async for chunk in response.aiter_bytes():
                yield chunk

    async def speech_to_text(
        self,
        audio_data: bytes,
        audio_filename: str,
        model_id: str = settings.SPEECHMATICS_STT_MODEL_ID,
        language_code: str | None = settings.SPEECHMATICS_STT_LANGUAGE,
    ) -> dict:
        if not self.api_key:
            raise ValueError("SPEECHMATICS_API_KEY is not set")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        config = {
            "type": "transcription",
            "transcription_config": {
                "model": model_id,
                "language": language_code or settings.SPEECHMATICS_STT_LANGUAGE,
            },
        }

        files = {
            "data_file": (audio_filename, audio_data, "audio/wav"),
            "config": (None, json.dumps(config), "application/json"),
        }

        # Submit transcription job
        response = await self._client.post(self.stt_base_url, headers=headers, files=files)
        if response.status_code not in (200, 201):
            logger.error("[SPEECHMATICS:STT] Failed to create job: body=%s", response.text[:300])
            raise httpx.HTTPStatusError(
                f"Speechmatics STT job returned {response.status_code}",
                request=response.request, response=response,
            )

        job_data = response.json()
        job_id = job_data["id"]

        # Poll until complete (max ~5 min)
        job_url = f"{self.stt_base_url}{job_id}"
        max_retries = 300
        poll_interval = 1.0

        for attempt in range(max_retries):
            await asyncio.sleep(poll_interval)
            status_res = await self._client.get(job_url, headers=headers)
            if status_res.status_code != 200:
                continue

            status_data = status_res.json()
            job_status = status_data.get("job", {}).get("status")

            if job_status == "done":
                break
            if job_status in ("rejected", "deleted", "expired", "failed"):
                raise RuntimeError(f"Speechmatics job {job_id} failed with status: {job_status}")
        else:
            raise RuntimeError(f"Speechmatics job {job_id} timed out after {max_retries * poll_interval:.0f}s")

        # Fetch transcript
        transcript_url = f"{self.stt_base_url}{job_id}/transcript"
        transcript_res = await self._client.get(transcript_url, headers=headers)
        transcript_res.raise_for_status()
        transcript_data = transcript_res.json()

        # Reconstruct text from results
        results = transcript_data.get("results", [])
        text_parts = []

        for item in results:
            alternatives = item.get("alternatives", [])
            if not alternatives:
                continue
            content = alternatives[0].get("content", "")
            if item.get("type") == "punctuation" and item.get("attaches_to") == "previous":
                if text_parts:
                    text_parts[-1] += content
                else:
                    text_parts.append(content)
            else:
                text_parts.append(content)

        final_text = " ".join(text_parts).strip()

        return {"text": final_text, "language_code": language_code}
