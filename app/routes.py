from fastapi import APIRouter, HTTPException
from app.schemas import PromptRequest
from app.services import extract_data_with_ai

router = APIRouter()

@router.post("/extract")
async def extract_itinerary_data(request: PromptRequest):
    try:
        extracted_data = extract_data_with_ai(request.prompt)
        return {"status": "success", "data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))