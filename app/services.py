import json
import google.generativeai as genai
from tavily import TavilyClient
from app.config import settings

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
    """Uses Tavily to search the web for real places matching the AI's criteria."""
    
    # Initialize the Tavily client
    tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    
    # Construct a highly specific Google search query
    query = f"Top rated {activity['type']} in {location} "
    if activity.get('specific_request') and activity['specific_request'] != "unspecified":
         query += f"serving {activity['specific_request']} "
    if activity.get('vibe') and activity['vibe'] != "unspecified":
         query += f"with a {activity['vibe']} vibe"
         
    print(f"Searching web for: {query}")
    
    # Execute the search
    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=3 # We only want the top 3 options
    )
    
    # Extract just the clean results
    return response.get("results", [])

def extract_specific_places_from_search(search_results: list, activity: dict) -> list:
    """Takes raw search results and uses Gemini to extract the actual place names."""
    
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
    
    Read the text above and extract the names of up to 3 SPECIFIC places (e.g., actual names of parks, cafes, restaurants). 
    Do not return aggregator names like "Tripadvisor" or "Justdial".
    
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