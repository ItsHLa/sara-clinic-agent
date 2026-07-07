from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    text: str = Field(min_length=1)
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"


class TTSStatus(BaseModel):
    status: str
    message: str | None = None


class WordTiming(BaseModel):
    text: str
    start: float
    end: float
    type: str


class STTResponse(BaseModel):
    model_config = {"from_attributes": True}

    text: str
    language_code: str | None = None
    language_probability: float | None = None
    words: list[WordTiming] | None = None
