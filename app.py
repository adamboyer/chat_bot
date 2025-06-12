from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging
import os
import uvicorn
from typing import List, Dict, Any, Optional

# -----------------------------------------------------------------------------
# ENV & LOGGING
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class Flight(BaseModel):
    id: str
    departure: str
    arrival: str
    departure_date: Optional[str] = None  # ISO‑8601 like "2025‑07‑01"
    arrival_date:   Optional[str] = None  # ISO‑8601 like "2025‑07‑01"
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
# Function‑tool stub – LLM implements body
# -----------------------------------------------------------------------------
@function_tool
def recommend_itinerary(input: ItineraryInput) -> Itinerary:
    """Planner logic REQUIRED (LLM‑only):
    1. When dates are flexible, pick the **most‑recent future departure_date** among `input.flights`.
    2. Always choose the **cheapest flight** that honours user constraints (e.g. cities, date, tickets).
    3. If the user expresses no strong hotel preference ("you pick", "any hotel", etc.), choose the **lowest night‑rate** hotel.
    4. Compute `total_cost` = flight.price + hotel.price_per_night × 3 (assume 3 nights unless the user states otherwise).
    5. Subtract `user_points / 100` dollars (1 point ≈ $0.01) from the total.  Set `points_used` accordingly.
    6. Return **only** a JSON object matching the `Itinerary` schema – no prose.
    """ 
    pass

# -----------------------------------------------------------------------------
# In‑memory chat sessions  { user_id: {agent, history (List[str])} }
# -----------------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    logger.info("Received request: %s", data)

    user_id  = data.get("user_id", "default")
    user_msg = data.get("message", "")
    flights  = data.get("flights", [])
    hotels   = data.get("hotels", [])
    points   = data.get("user_points", 0)

    tool_input = ItineraryInput(
        flights=[Flight(**f) for f in flights],
        hotels=[Hotel(**h) for h in hotels],
        user_points=points,
    )

    if user_id not in sessions:
        sessions[user_id] = {
            "agent": Agent(
                name="Itinerary Assistant",
                instructions=(
                    "You are **TripBot**, an expert travel‑planning assistant.

"
                    "Your job is to EFFICIENTLY gather missing info and then output a single JSON itinerary.
"
                    "Guidelines:
"
                    "• Ask brief clarification questions only if data is missing.
"
                    "• If the user lets you choose dates, pick the most‑recent future date.
"
                    "• If the user has no hotel preference, pick the cheapest hotel.
"
                    "• Always pick the cheapest flight that satisfies departure/arrival cities and chosen date.
"
                    "• After you have flights, hotels, and user_points, call `recommend_itinerary` and respond with **only** the JSON (no extra text).
"
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini",
            ),
            "history": [],
        }

    agent    = sessions[user_id]["agent"]
    history  = sessions[user_id]["history"]  # type: List[str]

    history_text = "\n".join(history + [user_msg]) if history else user_msg

    try:
        run_result = await Runner.run(agent, history_text)
        logger.info("Raw run_result.final_output: %s", run_result.final_output)

        # Save turns for context in the next call
        history.extend([user_msg, str(run_result.final_output)])

        # Try to parse structured itinerary
        try:
            itinerary = run_result.final_output_as(Itinerary)
            return JSONResponse(content=itinerary.dict())
        except Exception:
            return JSONResponse(content={"response": str(run_result.final_output)})

    except Exception as err:
        logger.exception("Runner failed: %s", err)
        return JSONResponse(content={"error": str(err)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
