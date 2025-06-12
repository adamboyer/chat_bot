from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging, json, uvicorn, os
from typing import List, Dict, Any, Optional

# -----------------------------------------------------------------------------
# ENV & LOGGING
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripbot")

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class Flight(BaseModel):
    id: str
    departure: str
    arrival: str
    departure_date: Optional[str] = None  # ISO‑8601 e.g. "2025‑07‑01"
    arrival_date:   Optional[str] = None  # ISO‑8601 e.g. "2025‑07‑01"
    price: float

class Hotel(BaseModel):
    id: str
    name: str
    price_per_night: float

class ItineraryInput(BaseModel):
    flights: List[Flight]
    hotels: List[Hotel]
    user_points: int

class Itinerary(BaseModel):
    flight: Flight
    hotel: Hotel
    total_cost: float
    points_used: int
    notes: str

# -----------------------------------------------------------------------------
# LLM Function‑Tool (the model implements the body)
# -----------------------------------------------------------------------------
@function_tool
def recommend_itinerary(input: ItineraryInput) -> Itinerary:
    """LLM‑only planning guidelines:
    1. If dates are flexible, pick the **nearest future** `departure_date`.
    2. Always choose the **cheapest** flight & hotel satisfying constraints.
    3. If no hotel preference, default to the lowest nightly rate.
    4. Assume 3 nights unless user specifies otherwise and compute `total_cost`.
    5. Offset cost with reward points (1 pt = $0.01) and set `points_used`.
    6. Return **only** JSON conforming to the `Itinerary` schema.
    7. If the user simply wants to *see available options* (flights/hotels) without booking, list the top‑5 cheapest items instead of an itinerary.
    """
    pass

# -----------------------------------------------------------------------------
# In‑memory chat sessions  { user_id: { agent, history (List[str]) } }
# -----------------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    logger.info("Request: %s", data)

    # ---------- Parse user payload ----------
    user_id   = data.get("user_id", "default")
    user_msg  = data.get("message", "")
    flights   = data.get("flights", [])
    hotels    = data.get("hotels",  [])
    points    = data.get("user_points", 0)

    # ---------- Create session on first turn ----------
    if user_id not in sessions:
        sessions[user_id] = {
            "agent": Agent(
                name="TripBot",
                instructions=(
                    "You are **TripBot**, an expert travel planner.\n"
                    "• If the user wants to view options, list the 5 cheapest flights or hotels from the JSON.\n"
                    "• Ask only for missing information.\n"
                    "• If user lets you pick dates, choose the nearest future date.\n"
                    "• If user has no hotel preference, pick the cheapest hotel.\n"
                    "• Always choose the cheapest valid flight.\n"
                    "• Once you have flights, hotels and points, call `recommend_itinerary` and respond **only** with the JSON itinerary."
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini",
            ),
            "history": []  # List[str]
        }

    agent   = sessions[user_id]["agent"]
    history = sessions[user_id]["history"]

    # ---------- Build conversation string ----------
    prior_text = "\n".join(history) if history else ""
    details_block = (
        f"FLIGHTS_JSON: {json.dumps(flights)}\n"
        f"HOTELS_JSON:  {json.dumps(hotels)}\n"
        f"USER_POINTS:  {points}"
    )
    conversation = "\n".join(filter(None, [prior_text, user_msg, details_block]))
    logger.info("Conversation sent to agent:\n%s", conversation)

    # ---------- Call the agent ----------
    try:
        run_result = await Runner.run(agent, conversation)
        assistant_reply = str(run_result.final_output)
        logger.info("LLM raw output: %s", assistant_reply)

        # Store turns (text only)
        history.extend([user_msg, assistant_reply])

        # Attempt to parse structured itinerary
        try:
            itinerary_dict = run_result.final_output_as(Itinerary).dict()
        except Exception:
            itinerary_dict = {}

        # Unified response: always include message + itinerary (may be empty)
        return JSONResponse(content={
            "message": assistant_reply,
            "itinerary": itinerary_dict
        })

    except Exception as e:
        logger.exception("Runner failed: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
