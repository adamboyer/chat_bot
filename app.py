from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner
import logging, json, uvicorn
from typing import Dict, Any

# ---------------------------------------------------------------------------
# ENV & LOGGING
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripbot")

# ---------------------------------------------------------------------------
# Pydantic model for the fabricated offer (human‑friendly fields)
# ---------------------------------------------------------------------------
class Offer(BaseModel):
    airline: str
    flight_number: str
    departure_time: str  # e.g. "2025‑07‑05T08:00"
    arrival_time:   str  # e.g. "2025‑07‑05T16:15"
    hotel_name: str
    total_cost: float
    notes: str

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
app = FastAPI()

# Per‑user session memory: stores an agent plus chat history
sessions: Dict[str, Dict[str, Any]] = {}

AGENT_PROMPT = (
    "You are **TripBot**, a friendly travel assistant.\n\n"
    "Ask follow‑up questions until you have: departure city, destination city, and reward‑points budget.\n"
    "When you have all three, invent a realistic cheap flight and hotel, then respond with JSON only (no markdown).\n\n"
    "Flight details to invent: airline, flight_number, departure_time, arrival_time.\n"
    "Hotel: hotel_name (no ID).\n"
    "Assume: flight_price = $250, hotel_price_per_night = $100, stay = 3 nights.\n"
    "total_cost = 250 + 3*100 = 550.\n"
    "Use user_points at $0.01/pt toward total_cost; mention points used inside notes if relevant.\n\n"
    "If you still need info →\n"
    "{\n  \"message\": \"I still need your departure city…\",\n  \"offer\": {}\n}\n\n"
    "If you have enough info →\n"
    "{\n  \"message\": \"Here is your offer!\",\n  \"offer\": {\n    \"airline\": \"BudgetAir\",\n    \"flight_number\": \"F123\",\n    \"departure_time\": \"2025‑07‑05T08:00\",\n    \"arrival_time\":   \"2025‑07‑05T16:15\",\n    \"hotel_name\":    \"Happy Stay Inn\",\n    \"total_cost\": 550,\n    \"notes\": \"Includes 3‑night stay\"\n  }\n}"
)

# ---------------------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    uid = data.get("user_id", "guest")
    user_msg = data.get("message", "")

    # Initialise session
    if uid not in sessions:
        sessions[uid] = {
            "agent": Agent(name="TripBot", instructions=AGENT_PROMPT, model="gpt-4o-mini"),
            "history": []
        }

    agent   = sessions[uid]["agent"]
    history = sessions[uid]["history"]

    conversation = "\n".join(history + [user_msg]) if history else user_msg
    result = await Runner.run(agent, conversation)
    assistant_reply = str(result.final_output)
    history.extend([user_msg, assistant_reply])

    try:
        payload = json.loads(assistant_reply)
        if "message" not in payload:
            raise ValueError("missing message key")
        if "offer" not in payload:
            payload["offer"] = {}
    except Exception as err:
        logger.warning("Non‑JSON assistant reply: %s", err)
        payload = {"message": assistant_reply, "offer": {}}

    return JSONResponse(payload)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
