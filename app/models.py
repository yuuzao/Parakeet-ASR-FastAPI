from pydantic import BaseModel

class Word(BaseModel):
    word: str
    start: float
    end: float
    score: float = 0.0
    start_offset: int
    end_offset: int

class Segment(BaseModel):
    start: float
    end: float
    start_offset: int
    end_offset: int
    text: str
    words: list[Word]

class Segments(BaseModel):
    segments: list[Segment]
    word_segments: list[Word]

class TranscriptionResponse(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: Segments

