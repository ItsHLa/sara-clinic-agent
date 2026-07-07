from fastapi import Request

from voice_agent.agent.base import AgentService
from voice_agent.services.base import VoiceService
from voice_agent.config import settings


def get_voice_service(request: Request):
    provider = request.headers.get("x-voice-provider", settings.VOICE_PROVIDER).lower()
    services = getattr(request.app.state, "voice_services", {})
    return services.get(provider, request.app.state.voice_service)


def get_agent_service(request: Request):
    return request.app.state.agent_service
