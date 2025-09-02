from pydantic import BaseModel
from typing import List
from .HighlightMoment import HighlightMoment


class HighlightMoments(BaseModel):
    highlights: List[HighlightMoment]