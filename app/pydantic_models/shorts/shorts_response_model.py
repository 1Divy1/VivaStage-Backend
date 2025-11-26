from pydantic import BaseModel
from typing import List
from .short_model import ShortModel


class ShortsResponseModel(BaseModel):
    highlights: List[ShortModel]