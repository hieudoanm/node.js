from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


# Request body model
class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"


@app.post("/chat")
async def chat(req: PromptRequest):
    try:
        response = openai.ChatCompletion.create(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
        )
        return {
            "prompt": req.prompt,
            "response": response.choices[0].message["content"],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/models")
async def list_models():
    """
    Fetches available models from OpenAI's API.
    """
    try:
        models = openai.Model.list()
        # Only return model IDs for simplicity
        model_ids = [m["id"] for m in models["data"]]
        return {"available_models": sorted(model_ids)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "FastAPI + OpenAI server is running"}
