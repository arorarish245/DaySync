from fastapi import APIRouter, HTTPException
from app.schemas import PromptRequest
from app.services import extract_data_with_ai, search_for_places, extract_specific_places_from_search, filter_by_distance, get_driving_time

router = APIRouter()

@router.post("/extract")
async def extract_itinerary_data(request: PromptRequest):
    try:
        extracted_data = extract_data_with_ai(request.prompt, request.user_city)
        location = extracted_data.get("location", "Bengaluru")
        
        enriched_activities = []
        previous_coords = None 
        
        for activity in extracted_data.get("activities", []):
            raw_search_results = search_for_places(location, activity)
            specific_places = extract_specific_places_from_search(raw_search_results, activity)
            
            # 1. Base Map Filter
            reality_checked_places = filter_by_distance(location, specific_places, max_radius_km=15.0)
            
            # 2. THE LOGISTICS ENGINE (With Failsafe)
            final_places = []
            backup_places = [] # NEW: Store places that fail the time check
            
            for place in reality_checked_places:
                if not place.get("map_verified"):
                    final_places.append(place)
                    continue
                    
                if previous_coords:
                    driving_time_mins = get_driving_time(previous_coords, place["coordinates"])
                    
                    if driving_time_mins is not None:
                        place["driving_time_from_previous_mins"] = driving_time_mins
                        
                        if driving_time_mins > 25.0:
                            print(f"ROUTE TOO LONG: {place['name']} takes {driving_time_mins} mins.")
                            backup_places.append(place) # Put it in timeout, don't delete it
                            continue 
                        else:
                            print(f"ROUTE VERIFIED: {place['name']} is a {driving_time_mins} min drive.")
                            final_places.append(place)
                    else:
                        final_places.append(place) # If OSRM fails, keep it
                else:
                    final_places.append(place) # First activity has no previous drive
            
            # --- THE FAILSAFE ---
            if not final_places and backup_places:
                print(f"FAILSAFE TRIGGERED for {activity['type']}! All places were >25 mins. Returning the closest option.")
                backup_places.sort(key=lambda x: x.get("driving_time_from_previous_mins", 999))
                final_places.append(backup_places[0])
            
            # --- THE FIX: Update the "Car" location for the next loop ---
            # Search through the list to find the first place with real GPS coordinates
            previous_coords = None 
            for place in final_places:
                if place.get("map_verified") and "coordinates" in place:
                    previous_coords = place["coordinates"]
                    break  # We found our starting point! Stop looking.
                
            activity["suggested_places"] = final_places
            enriched_activities.append(activity)
            
        extracted_data["activities"] = enriched_activities
        
        return {"status": "success", "data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))