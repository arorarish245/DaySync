import json
import google.generativeai as genai
from app.config import settings

# Configure the API globally for this service
genai.configure(api_key=settings.GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an expert data extraction algorithm. Your job is to extract EVERY activity, errand, and task from a user's prompt and output strictly in valid JSON format.
Do not include markdown formatting, conversational text, or explanations. Just return the raw JSON object.

CRITICAL INSTRUCTIONS:
1. You must extract EVERY single location mentioned, including errands (like repair shops, grocery stores) and leisure (like bowling, movies, pubs).
2. Order the activities chronologically.

Follow this exact JSON schema:
{
  "location": "City, Neighborhood",
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

def extract_data_with_ai(user_prompt: str) -> dict:
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Prompt: {user_prompt}"
    response = model.generate_content(full_prompt)
    
    return json.loads(response.text)