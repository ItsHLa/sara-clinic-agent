from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8")

    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_BASE_URL: str = "https://api.elevenlabs.io"
    ELEVENLABS_VOICE_ID: str = "JBFqnCBsd6RMkjVDRZzb"
    ELEVENLABS_TTS_MODEL_ID: str = "eleven_multilingual_v2"
    ELEVENLABS_STT_MODEL_ID: str = "scribe_v2"
    SPEECHMATICS_API_KEY: str = ""
    SPEECHMATICS_TTS_BASE_URL: str = "https://preview.tts.speechmatics.com"
    SPEECHMATICS_STT_BASE_URL: str = "https://asr.api.speechmatics.com/v2/jobs/"
    SPEECHMATICS_TTS_VOICE_ID: str = "sarah"
    SPEECHMATICS_STT_MODEL_ID: str = "melia-1"
    SPEECHMATICS_STT_LANGUAGE: str = "multi"
    VOICE_PROVIDER: str = "elevenlabs"
    AGENT_URL: str = ""
    AGENT_INTERNAL_TOKEN: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
