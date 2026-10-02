from fastapi import FastAPI, HTTPException

from app.schema.device import (
    ExtractRequest,
    ExtractResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from app.services.extraction import (
    extract_device
)

from app.services.search import (
    hybrid_search
)


app = FastAPI(
    title="Mobile Device Assistant API",
    description=(
        "API for mobile-device specification extraction "
        "and hybrid BM25 + dense search."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Mobile Device Assistant API"
    }


@app.post("/extract", response_model=ExtractResponse)

def extract( request: ExtractRequest ):

    try:

        device = extract_device(
            request.text
        )

        return device

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post( "/search",response_model=SearchResponse)

def search( request: SearchRequest ):

    try:

        results = hybrid_search(
            request.q,
            k=3
        )

        return SearchResponse(
            results=[
                SearchResult(
                    document=document,
                    score=score
                )
                for document, score in results
            ]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )