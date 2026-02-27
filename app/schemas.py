from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    user_city: str | None = "Bengaluru, India"