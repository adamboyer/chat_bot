from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging
import os
import uvicorn
from typing import List, Dict, Any

# ------------------------------------------------------------------
# ENV & LOGGING
# ------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Pydantic models for strict schemas
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Tool definition – LLM fills the body
# ------------------------------------------------------------------
@function_tool
def recommend_itinerary(input: ItineraryInput) -> Itinerary:
    """Select the best flight/hotel based on price & points and return an itinerary."""
    pass  # LLM will implement

# ------------------------------------------------------------------
# Server‑side conversation memory (very simple)
# user_id -> {"agent": Agent, "history": List[Any] }
# ------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        logger.info("Received request: %s", data)

        user_id: str = data.get("user_id", "default")
        user_message: str = data.get("message", "")
        flights = data.get("flights", [])
        hotels = data.get("hotels", [])
        user_points = data.get("user_points", 0)

        # Convert raw lists to Pydantic objects
        tool_input = ItineraryInput(
            flights=[Flight(**f) for f in flights],
            hotels=[Hotel(**h) for h in hotels],
            user_points=user_points,
        )

        # Initialise session if first turn
        if user_id not in sessions:
            agent = Agent(
                name="Itinerary Assistant",
                instructions=(
                    "You are a helpful travel planner. Ask for any missing data (flights, hotels, points). "
                    "When you have everything, call the `recommend_itinerary` tool to reply in JSON."
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini",
            )
            sessions[user_id] = {"agent": agent, "history": []}

        session = sessions[user_id]
        history: List[Any] = session["history"]
        agent: Agent = session["agent"]

        # Build the full conversation to preserve context
        full_inputs = history + [user_message, tool_input]

        logger.info("Calling Runner.run with %d messages", len(full_inputs))
        run_result = await Runner.run(agent, *full_inputs)

        # Save user_msg and assistant reply for next turn
        history.extend([user_message, run_result.final_output])

        # Attempt to parse structured output
        try:
            itinerary = run_result.final_output_as(Itinerary)
            return JSONResponse(content=itinerary.dict())
        except Exception:
            return JSONResponse(content={"response": str(run_result.final_output)})

    except Exception as err:
        logger.exception("Unhandled error")
        return JSONResponse(content={"error": str(err)}, status_code=500)

# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
