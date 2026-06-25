from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voice_agent.config import settings

DEFAULT_VOICE_ID = settings.ELEVENLABS_VOICE_ID
DEFAULT_MODEL_ID = settings.ELEVENLABS_TTS_MODEL_ID


class AgentEvent:
    __slots__ = ("event", "data")

    def __init__(self, event, data):
        self.event = event
        self.data = data


class AgentService(ABC):
    @abstractmethod
    async def chat(self, question, session_id=None):
        ...
