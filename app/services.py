import json
import urllib.parse
import google.generativeai as genai
from tavily import TavilyClient
from app.config import settings
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

# Configure the API globally for this service
genai.configure(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an expert data extraction algorithm. Your job is to extract EVERY activity, errand, and task from a user's prompt and output strictly in valid JSON format.
Do not include markdown formatting, conversational text, or explanations. Just return the raw JSON object.

CRITICAL INSTRUCTIONS:
1. GEOGRAPHIC CONTEXT: Pay attention to the "User's Current/Target City". If they mention a neighborhood without a city, append the user_city UNLESS the neighborhood is globally famous and clearly belongs to a different major city
 (e.g., if user_city is Bengaluru but they ask for Bandra, output "Bandra, Mumbai, India").
2. You must extract EVERY single location mentioned, including errands (like repair shops, grocery stores) and leisure (like bowling, movies, pubs).
3. Order the activities chronologically. 

Follow this exact JSON schema:
{
  "location": "Neighborhood, City, State, Country",
  "budget": "low|medium|high|unspecified",
  "activities": [
    {
      "type": "Describe the place type (e.g., cafe, electronics repair shop, bowling alley, pub)", 
      "specific_request": "Any specific item mentioned (e.g., vegan lunch, pizza, laptop fix)",
      "vibe": "quiet/loud/relaxing/unspecified", 
      "preferred_time": "morning/afternoon/evening/specific time"
    }
  ]
}
"""

def extract_data_with_ai(user_prompt: str, user_city: str) -> dict:
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser's Current/Target City: {user_city}\n\nUser Prompt: {user_prompt}"
    response = model.generate_content(full_prompt)
    
    return json.loads(response.text)


def search_for_places(location: str, activity: dict) -> list:
    """Uses Tavily to search the web for real FAMOUS places matching the AI's criteria."""
    
    # Initialize the Tavily client
    tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    
    # Construct a highly specific Google search query
    query = f"Most famous, iconic, and top-rated {activity['type']} in {location} "
    if activity.get('specific_request') and activity['specific_request'] != "unspecified":
         query += f"serving {activity['specific_request']} "
    if activity.get('vibe') and activity['vibe'] != "unspecified":
         query += f"with a {activity['vibe']} vibe"
         
    print(f"Searching web for: {query}")
    
    # Execute the search
    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=4 
    )
    
    # Extract just the clean results
    return response.get("results", [])

def extract_specific_places_from_search(search_results: list, activity: dict) -> list:
    """Takes raw search results and Uses AI to extract ONLY widely recognized, famous places, with strict enterprise-level quality control."""
    
    # If the search failed, return an empty list
    if not search_results:
        return []

    # Combine all the search content into one big text block
    search_context = "\n".join([f"Title: {res['title']}\nContent: {res['content']}" for res in search_results])
    
    # Prompt Gemini to read the text and pull out 3 specific places
    extraction_prompt = f"""
    You are an AI assistant helping to build an itinerary. I searched the web for a {activity['type']} matching the vibe "{activity.get('vibe', 'any')}".
    Here are the search results:
    
    {search_context}
    
    CRITICAL INSTRUCTIONS:
    1. Extract up to 3 SPECIFIC places (actual names of businesses).
    2. QUALITY CONTROL: You MUST strictly extract only highly famous, and widely recognized places. Reject any obscure stalls, unknown locations, or places with few reviews.
    3. Do not return aggregator names like "Tripadvisor" or "Justdial".
    
    Return strictly in this JSON format:
    [
      {{"name": "Specific Place Name", "description": "1 sentence explaining why it fits based on the search text"}}
    ]
    """
    
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(extraction_prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Failed to extract specific places: {e}")
        return []

# Initialize the map tool (OpenStreetMap requires a custom user_agent name)
geolocator = Nominatim(user_agent="DaySync_Local_Logistics_App")

def get_coordinates(place_name: str):
    """Turns a text address into GPS coordinates."""
    try:
        # Ask the map for the location
        location = geolocator.geocode(place_name, timeout=10)
        if location:
            return (location.latitude, location.longitude)
        return None
    except Exception as e:
        return None

def filter_by_distance(base_location: str, suggested_places: list, max_radius_km: float = 15.0) -> list:
    """Safely verifies places using an expanding geographic search to get distance in km."""
    base_coords = get_coordinates(base_location)
    
    if not base_coords:
        return suggested_places 

    valid_places = []
    
    # Break down the location into Neighborhood, City, State
    # e.g., ["Gandhi Nagar", "Ahmedabad", "Gujarat", "India"]
    location_parts = [p.strip() for p in base_location.split(',')]
    neighborhood = location_parts[0] if len(location_parts) > 0 else base_location
    broad_city = location_parts[1] if len(location_parts) > 1 else neighborhood
    state = location_parts[2] if len(location_parts) > 2 else broad_city

    for place in suggested_places:
        place_coords = None
        
        # SEARCH ATTEMPT 1: Exact Neighborhood (e.g., "Manek Chowk, Gandhi Nagar")
        place_coords = get_coordinates(f"{place['name']}, {neighborhood}")
        
        # SEARCH ATTEMPT 2: Broader City (e.g., "Manek Chowk, Ahmedabad")
        if not place_coords and broad_city != neighborhood:
            print(f"Expanding search for {place['name']} to {broad_city}...")
            place_coords = get_coordinates(f"{place['name']}, {broad_city}")
            time.sleep(1) # Extra sleep to avoid OpenStreetMap rate limits
            
        # SEARCH ATTEMPT 3: Whole State (e.g., "Manek Chowk, Gujarat")
        if not place_coords and state != broad_city:
            print(f"Expanding search for {place['name']} to {state}...")
            place_coords = get_coordinates(f"{place['name']}, {state}")
            time.sleep(1)

        # If we finally found the coordinates, calculate the distance!
        if place_coords:
            distance = geodesic(base_coords, place_coords).kilometers
            
            if distance <= max_radius_km:
                place["coordinates"] = {"lat": place_coords[0], "lng": place_coords[1]}
                place["distance_from_base_km"] = round(distance, 2)
                place["map_verified"] = True
                
                # Create a Google Maps route URL using exact coordinates!
                place["google_maps_url"] = f"https://www.google.com/maps/dir/?api=1&destination={place_coords[0]},{place_coords[1]}"
                
                valid_places.append(place)
                print(f"FULL VERIFIED: {place['name']} is {distance:.2f} km away.")
            else:
                print(f"TRASHED (Too Far): {place['name']} is {distance:.2f} km away.")
        else:
            print(f"WEB ONLY: Keeping {place['name']} without exact distance.")
            place["map_verified"] = False
            
            # Create a Google Maps Search URL using the name of the place!
            encoded_query = urllib.parse.quote(f"{place['name']}, {broad_city}")
            place["google_maps_url"] = f"https://www.google.com/maps/search/?api=1&query={encoded_query}"
            
            valid_places.append(place)
            
        time.sleep(1) 
        
    return valid_places