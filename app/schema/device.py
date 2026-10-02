from typing import Dict, Literal, List

from pydantic import BaseModel, Field, ConfigDict


class Device(BaseModel):
    brand: str
    model: str
    specs: Dict[str, str]
    release_year: int
    price_tier: Literal["budget", "mid-range", "flagship"]


class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=500)


class ExtractResponse(Device):
    pass


class SearchRequest(BaseModel):
    q: str = Field(..., max_length=500)


class SearchResult(BaseModel):
    document: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]