from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner
import logging, json, uvicorn
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# ENV & LOGGING
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripbot")

# ---------------------------------------------------------------------------
# Minimal data model for an offer
# ---------------------------------------------------------------------------
class Offer(BaseModel):
    flight_id: str
    hotel_id: str
    total_cost: float
    notes: str

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
app = FastAPI()

# One lightweight agent per session
sessions: Dict[str, Agent] = {}

AGENT_PROMPT = """
You are TripBot, a friendly travel assistant.

• If the user message doesn’t give a **destination city**, **departure city**, or **points budget**, ask concise questions to obtain the missing piece(s).
• When you have enough info (departure + destination + points), invent a cheap flight (id F123) and hotel (id H456) and calculate:
    total_cost = flight_price ($250) + hotel_price_per_night ($100) * 3 nights = $550
    points_used = min(points_budget, total_cost / 0.01)
    out_of_pocket = total_cost – points_used*0.01
• Respond with **ONLY** a JSON payload like:
{
  "message": "Here is your offer!",
  "offer": {
    "flight_id": "F123",
    "hotel_id": "H456",
    "total_cost": 550,
    "notes": "Includes 3‑night stay"
  }
}
If you still need data, respond with:
{
  "message": "I still need your points budget…",
  "offer": {}
}
No additional keys, no markdown fences.
"""

# ---------------------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    uid  = data.get("user_id", "guest")
    msg  = data.get("message", "")

    if uid not in sessions:
        sessions[uid] = Agent(name="TripBot", instructions=AGENT_PROMPT, model="gpt-4o-mini")

    agent = sessions[uid]
    result = await Runner.run(agent, msg)

    # The agent has been instructed to always return valid JSON
    try:
        payload = json.loads(str(result.final_output))
    except Exception as e:
        logger.warning("Agent returned non‑JSON: %s", e)
        payload = {"message": str(result.final_output), "offer": {}}

    # ensure both keys exist
    if "offer" not in payload:
        payload["offer"] = {}
    if "message" not in payload:
        payload["message"] = str(result.final_output)

    return JSONResponse(payload)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
