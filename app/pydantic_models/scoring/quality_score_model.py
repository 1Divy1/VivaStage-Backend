from pydantic import BaseModel, Field

class QualityScoreModel(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
