from pydantic import BaseModel, HttpUrl, field_validator


'''Reel model (Pydantic)'''


class GenerateShortsRequestModel(BaseModel):
    youtube_url: HttpUrl
    number_of_reels: int
    min_seconds: int
    max_seconds: int
    captions: bool = False
    language: str

    # Validate that the YouTube URL is valid and belongs to YouTube
    @field_validator('youtube_url')
    @classmethod
    def must_be_youtube(cls, v: HttpUrl) -> HttpUrl:
        allowed_domains = ["youtube.com", "www.youtube.com", "youtu.be"]
        domain = v.host.lower()

        if domain not in allowed_domains:
            raise ValueError("URL must be a YouTube link")

        return v
