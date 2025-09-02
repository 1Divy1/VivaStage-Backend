from pydantic import BaseModel


class HighlightMoment(BaseModel):
    start: float
    end: float
    text: str
    reason: str
