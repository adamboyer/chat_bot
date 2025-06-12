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
# Pydantic model for the fabricated offer
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

# Per‑user session memory: stores an agent plus chat history
sessions: Dict[str, Dict[str, Any]] = {}

AGENT_PROMPT = (
    "You are **TripBot**, a friendly travel assistant.\n\n"
    "• If the user message is missing any of the following, ask concise follow‑up questions:\n"
    "    – departure city\n    – destination city\n    – reward‑points budget\n\n"
    "• Once you have all three, invent an economical flight (ID `F123`) and hotel (ID `H456`).\n"
    "  Assume: flight_price = $250, hotel_price_per_night = $100, stay = 3 nights.\n"
    "  total_cost = $250 + 3·$100 = $550.\n"
    "  points_used = min(points_budget, total_cost / 0.01).\n\n"
    "• Respond **only** with JSON — NO markdown fences — using one of the two schemas below.\n\n"
    "If you still need info →\n"
    "{\n  \"message\": \"I still need your destination city…\",\n  \"offer\": {}\n}\n\n"
    "If you have enough info →\n"
    "{\n  \"message\": \"Here is your offer!\",\n  \"offer\": {\n    \"flight_id\": \"F123\",\n    \"hotel_id\":  \"H456\",\n    \"total_cost\": 550,\n    \"notes\": \"Includes 3‑night stay\"\n  }\n}"
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
            "history": []  # alternating user / assistant messages
        }

    agent = sessions[uid]["agent"]
    history = sessions[uid]["history"]

    # Build conversation text with memory
    conversation = "\n".join(history + [user_msg]) if history else user_msg

    # Call the LLM agent
    result = await Runner.run(agent, conversation)
    assistant_reply = str(result.final_output)

    # Update history
    history.extend([user_msg, assistant_reply])

    # Parse the JSON payload TripBot should have returned
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
