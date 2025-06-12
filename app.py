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
    """From the provided flights and hotels:
    • If the user gave no specific date, list the available dates for each flight.
    • If the user has no hotel preference, pick the **cheapest hotel**.
    • Always prefer the cheapest valid flight & hotel and use `user_points` to offset cost.
    Return a JSON itinerary conforming to the `Itinerary` model."""
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
                    "You are a travel‑planning assistant.\n"
                    "• Ask for missing flights/hotels/points.\n"
                    "• If the user lets you pick dates, choose the most‑recent flight date.\n"
                    "• If the user has no hotel preference, choose the cheapest hotel.\n"
                    "• Once everything is available, call `recommend_itinerary` and reply with the JSON only."
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
