from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="DaySync", version="1.0.0")

# Include the routes from our routes.py file
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the DaySync API. Go to /docs to test."}