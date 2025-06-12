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
    • If the user gave no specific date, treat "pick any date" as **pick the most recent available option**.
    • Always prefer the **cheapest** flight and hotel that satisfy any user constraints.
    • Use user_points to offset cost where possible.
    Return the chosen itinerary as structured JSON conforming to the `Itinerary` model."""
    pass


# ----------------------------------------------------------------------------
# Simple in‑memory sessions {user_id: {agent, history}}
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

    # initialise session
    if user_id not in sessions:
        sessions[user_id] = {
            "agent": Agent(
                name="Itinerary Assistant",
                instructions=(
                    "You are a helpful travel planner. Ask for any missing data (flights, hotels, points). "
                    "When you have everything, call the `recommend_itinerary` tool to reply in JSON."
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini",
            ),
            "history": [],  # list[str]
        }

    agent = sessions[user_id]["agent"]
    history: List[str] = sessions[user_id]["history"]

    # Compose a single text block with prior turns for context
    history_text = "\n".join(history + [user_msg]) if history else user_msg

    try:
        # Runner.run expects exactly: (agent, *messages)
        run_result = await Runner.run(agent, history_text)

        # store turns for next round (only text, not tool dict)
        history.append(user_msg)
        history.append(str(run_result.final_output))

        # try structured parse
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
