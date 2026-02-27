from fastapi import APIRouter, HTTPException
from app.schemas import PromptRequest
from app.services import extract_data_with_ai, search_for_places, extract_specific_places_from_search

router = APIRouter()

@router.post("/extract")
async def extract_itinerary_data(request: PromptRequest):
    try:
        extracted_data = extract_data_with_ai(request.prompt, request.user_city)
        location = extracted_data.get("location", "Bengaluru")
        enriched_activities = []
        
        for activity in extracted_data.get("activities", []):
            # 1. Search the web (returns Tripadvisor/Justdial links)
            raw_search_results = search_for_places(location, activity)
            
            # 2. Use the AI to read those links and pull out specific place names!
            specific_places = extract_specific_places_from_search(raw_search_results, activity)
            
            # 3. Add the clean specific places to the data
            activity["suggested_places"] = specific_places
            enriched_activities.append(activity)
            
        extracted_data["activities"] = enriched_activities
        
        return {"status": "success", "data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))