import json
import logging
from collections.abc import AsyncIterator

import httpx
from httpx_sse import aconnect_sse, SSEError

from voice_agent.agent.base import AgentEvent, AgentService
from voice_agent.config import settings

logger = logging.getLogger("voice_agent.chat_agent")


class ChatAgent(AgentService):
    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
    ) -> None:
        self.url = url or settings.AGENT_URL
        self.token = token or settings.AGENT_INTERNAL_TOKEN
        self._client = httpx.AsyncClient()
        logger.info("[CHAT_AGENT] Initialized url=%s has_token=%s", self.url, bool(self.token))

    async def chat(
        self,
        question: str,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        payload = {
            "question": question,
            "streaming": True,
            "overrideConfig": {"sessionId": session_id},
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token.strip()}"

        try:
            async with aconnect_sse(
                self._client, "POST", self.url, headers=headers, json=payload, timeout=60.0
            ) as event_source:
                async for event in event_source.aiter_sse():
                    try:
                        chunk = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue

                    event_type = chunk.get("event")
                    if event_type not in ("token", "error"):
                        continue

                    data = chunk.get("data", "")
                    if not isinstance(data, str):
                        data = json.dumps(data)

                    if event_type == "error":
                        yield AgentEvent(event="error", data=data)
                        return

                    yield AgentEvent(event="token", data=data)

        except SSEError as e:
            logger.error("[CHAT_AGENT] SSE failed: %s", e)
            yield AgentEvent(event="error", data=f"Agent server error: {e}")
