from pydantic import BaseModel


class ShortModel(BaseModel):
    start: float
    end: float
    text: str
    reason: str
