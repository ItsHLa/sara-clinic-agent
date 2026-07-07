import asyncio
import base64
import json
import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Request
from fastapi.responses import StreamingResponse

from voice_agent.agent.base import AgentService
from voice_agent.agent.schemas import AgentChatRequest
from voice_agent.deps import get_agent_service, get_voice_service
from voice_agent.services.base import VoiceService

logger = logging.getLogger("voice_agent.pipeline")

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat")
async def agent_chat(
    body: AgentChatRequest,
    agent: Annotated[AgentService, Depends(get_agent_service)],
    request: Request,
):
    logger.info("[CHAT] === STEP 1: Chat request received === question=%s session=%s",
                body.question[:50] if body.question else "", body.session_id)

    async def event_stream():
        step = 0
        async for event in agent.chat(question=body.question, session_id=body.session_id):
            step += 1
            logger.info("[CHAT]   STEP 2.%d: agent event=%s data_len=%d", step, event.event, len(event.data))
            yield f"event: {event.event}\ndata: {event.data}\n\n"

    logger.info("[CHAT] === Returning SSE stream ===")
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/pipeline")
async def audio_pipeline(
    file: Annotated[UploadFile, File(description="Audio file to transcribe and process")],
    voice: Annotated[VoiceService, Depends(get_voice_service)],
    agent: Annotated[AgentService, Depends(get_agent_service)],
    request: Request,
    session_id: str | None = Form(None),
    language_code: str | None = Form(None),
):
    logger.info("[PIPELINE] === STEP 1: Request received === file=%s session=%s",
                file.filename, session_id)

    async def event_stream():
        audio_data = await file.read()
        logger.info("[PIPELINE]   STEP 2: Audio read: %d bytes", len(audio_data))

        logger.info("[PIPELINE]   STEP 3: Calling STT (speech_to_text)...")
        transcription = await voice.speech_to_text(
            audio_data=audio_data,
            audio_filename=file.filename or "audio.wav",
            language_code=language_code,
        )
        question = transcription["text"]
        logger.info("[PIPELINE]   STEP 4: STT complete. text='%s'", question[:80])

        yield f"event: transcript\ndata: {json.dumps({'text': question, 'language_code': transcription.get('language_code')})}\n\n"

        step = 0
        logger.info("[PIPELINE]   STEP 5: Calling Agent chat...")
        async for event in agent.chat(question=question, session_id=session_id):
            step += 1
            logger.info("[PIPELINE]   STEP 5.%d: agent event=%s", step, event.event)
            yield f"event: {event.event}\ndata: {event.data}\n\n"

        logger.info("[PIPELINE] === Pipeline complete ===")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_SENTENCE_RE = re.compile(r".*?[.!?](?=\s|$)")


def _iter_sentences(text):
    sentences = []
    pos = 0
    while pos < len(text):
        m = _SENTENCE_RE.match(text, pos)
        if m:
            sentences.append(m.group().strip())
            pos = m.end()
        else:
            break
    return sentences, text[pos:].strip()


@router.post("/full-cycle")
async def full_cycle(
    file: Annotated[UploadFile, File(description="Audio file to transcribe and respond to")],
    voice: Annotated[VoiceService, Depends(get_voice_service)],
    agent: Annotated[AgentService, Depends(get_agent_service)],
    request: Request,
    session_id: str | None = Form(None),
    language_code: str | None = Form(None),
    voice_id: str = Form("JBFqnCBsd6RMkjVDRZzb"),
):
    logger.info("[FULL-CYCLE] === STEP 1: Request received === file=%s session=%s", file.filename, session_id)

    audio_data = await file.read()
    audio_filename = file.filename or "audio.wav"
    logger.info("[FULL-CYCLE]   STEP 2: Audio read: %d bytes", len(audio_data))

    async def event_stream():
        logger.info("[FULL-CYCLE]   STEP 3: Calling STT (speech_to_text)...")
        transcription = await voice.speech_to_text(
            audio_data=audio_data,
            audio_filename=audio_filename,
            language_code=language_code,
        )
        question = transcription["text"]
        logger.info("[FULL-CYCLE]   STEP 4: STT complete. text='%s'", question[:80])
        yield f"event: transcript\ndata: {json.dumps({'text': question})}\n\n"

        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        audio_queue: asyncio.Queue[tuple | None] = asyncio.Queue()
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        agent_error = [None]

        async def collect_tokens():
            buffer = ""
            async for event in agent.chat(question=question, session_id=session_id):
                if event.event == "error":
                    agent_error[0] = event.data
                    await sentence_queue.put(None)
                    await text_queue.put(None)
                    return
                await text_queue.put(event.data)
                buffer += event.data
                done, rest = _iter_sentences(buffer)
                for s in done:
                    await sentence_queue.put(s)
                buffer = rest
            if buffer.strip():
                await sentence_queue.put(buffer.strip())
            await sentence_queue.put(None)
            await text_queue.put(None)

        async def tts_worker():
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    await audio_queue.put(None)
                    return
                logger.info("[FULL-CYCLE]   TTS sentence: '%s'", sentence[:60])
                await audio_queue.put(("sentence", sentence))
                async for chunk in voice.text_to_speech(text=sentence, voice_id=voice_id):
                    b64 = base64.b64encode(chunk).decode()
                    await audio_queue.put(("audio", b64))
                await audio_queue.put(("audio_end", None))

        collect_task = asyncio.create_task(collect_tokens())
        tts_task = asyncio.create_task(tts_worker())

        text_done = False
        audio_done = False
        accumulated_text = ""

        while not (text_done and audio_done):
            while not text_done:
                try:
                    token = text_queue.get_nowait()
                    if token is None:
                        text_done = True
                        yield f"event: response_done\ndata: {json.dumps(accumulated_text)}\n\n"
                    else:
                        accumulated_text += token
                        yield f"event: token\ndata: {json.dumps(token)}\n\n"
                except asyncio.QueueEmpty:
                    break

            while not audio_done:
                try:
                    item = audio_queue.get_nowait()
                    if item is None:
                        audio_done = True
                    else:
                        typ, val = item
                        if typ == "sentence":
                            yield f"event: sentence\ndata: {json.dumps(val)}\n\n"
                        elif typ == "audio":
                            yield f"event: audio\ndata: {val}\n\n"
                        elif typ == "audio_end":
                            yield "event: audio_end\ndata: \n\n"
                except asyncio.QueueEmpty:
                    break

            if agent_error[0]:
                yield f"event: error\ndata: {json.dumps(agent_error[0])}\n\n"
                return

            if not (text_done and audio_done):
                await asyncio.sleep(0.02)

        logger.info("[FULL-CYCLE] === Done ===")
        yield "event: done\ndata: \n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/voice-cycle")
async def voice_cycle(
    file: Annotated[UploadFile, File(description="Audio file to transcribe and respond to")],
    voice: Annotated[VoiceService, Depends(get_voice_service)],
    agent: Annotated[AgentService, Depends(get_agent_service)],
    request: Request,
    session_id: str | None = Form(None),
    language_code: str | None = Form(None),
    voice_id: str = Form("JBFqnCBsd6RMkjVDRZzb"),
):
    logger.info("[VOICE-CYCLE] === STEP 1: Request received === file=%s session=%s", file.filename, session_id)

    async def audio_stream():
        audio_data = await file.read()
        logger.info("[VOICE-CYCLE]   STEP 2: Audio read: %d bytes", len(audio_data))

        logger.info("[VOICE-CYCLE]   STEP 3: Calling STT (speech_to_text)...")
        transcription = await voice.speech_to_text(
            audio_data=audio_data,
            audio_filename=file.filename or "audio.wav",
            language_code=language_code,
        )
        question = transcription["text"]
        logger.info("[VOICE-CYCLE]   STEP 4: STT complete. text='%s'", question[:80])

        response_parts = []
        step = 0
        logger.info("[VOICE-CYCLE]   STEP 5: Calling Agent chat...")
        async for event in agent.chat(question=question, session_id=session_id):
            step += 1
            if event.event == "error":
                logger.error("[VOICE-CYCLE]   STEP 5.%d: AGENT ERROR: %s", step, event.data)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Agent error: {event.data}",
                )
            logger.info("[VOICE-CYCLE]   STEP 5.%d: agent event=%s", step, event.event)
            response_parts.append(event.data)

        agent_text = " ".join(response_parts).strip()
        logger.info("[VOICE-CYCLE]   STEP 6: Agent response complete. text='%s'", agent_text[:80])
        if not agent_text:
            logger.warning("[VOICE-CYCLE]   STEP 6: Empty agent response, returning")
            return

        chunk_count = 0
        logger.info("[VOICE-CYCLE]   STEP 7: Calling TTS (text_to_speech)...")
        async for chunk in voice.text_to_speech(text=agent_text, voice_id=voice_id):
            chunk_count += 1
            yield chunk
        logger.info("[VOICE-CYCLE]   STEP 8: TTS complete. %d chunks sent", chunk_count)

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline"},
    )
