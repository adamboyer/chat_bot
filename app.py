from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging
import os
import uvicorn
from typing import List, Dict, Any

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------
class Flight(BaseModel):
    id: str
    departure: str
    arrival: str
    departure_date: str  # ISO‑8601 like "2025-07-01"
    arrival_date: str    # ISO‑8601 like "2025-07-01"
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

# ----------------------------------------------------------------------------
# Tool – LLM fills body
# ----------------------------------------------------------------------------
@function_tool
def recommend_itinerary(input: ItineraryInput) -> Itinerary:
    """From the provided flights and hotels:
    • If the user gave no specific date, treat "pick any date" as **choose the most recent available flight date**.
    • If the user has no hotel preference (e.g. "you pick the hotel" or similar), pick the **cheapest hotel**.
    • Always prefer the **cheapest** flight and hotel that satisfy any explicit user constraints.
    • Use `user_points` to offset cost where possible.
    Return the chosen itinerary as structured JSON conforming to the `Itinerary` model."""
    pass  # Implemented by the LLM

# ----------------------------------------------------------------------------
# In‑memory sessions {user_id: {agent, history}}
# ----------------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    logger.info("Received request: %s", data)

    user_id = data.get("user_id", "default")
    user_msg = data.get("message", "")
    flights = data.get("flights", [])
    hotels = data.get("hotels", [])
    user_points = data.get("user_points", 0)

    tool_input = ItineraryInput(
        flights=[Flight(**f) for f in flights],
        hotels=[Hotel(**h) for h in hotels],
        user_points=user_points,
    )

    if user_id not in sessions:
        sessions[user_id] = {
            "agent": Agent(
                name="Itinerary Assistant",
                instructions=(
                    "You are a travel‑planning assistant.\n"
                    "• Ask for any missing data (flights, hotels, points).\n"
                    "• If the user says you may choose dates, pick the **most recent** flight date.\n"
                    "• If the user says you may choose the hotel or has no preference, pick the **cheapest** hotel.\n"
                    "• Always select the cheapest flight that meets constraints.\n"
                    "• Once you have flights, hotels, and points, call `recommend_itinerary` and respond **only** with the JSON itinerary."
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini",
            ),
            "history": [],
        }

    agent = sessions[user_id]["agent"]
    history: List[str] = sessions[user_id]["history"]

    history_text = "\n".join(history + [user_msg]) if history else user_msg

    try:
        run_result = await Runner.run(agent, history_text, tool_input)
        history.extend([user_msg, str(run_result.final_output)])

        try:
            itinerary = run_result.final_output_as(Itinerary)
            return JSONResponse(content=itinerary.dict())
        except Exception:
            return JSONResponse(content={"response": str(run_result.final_output)})

    except Exception as e:
        logger.exception("Runner failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
