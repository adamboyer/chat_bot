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
# LLM function‑tool (body filled by the model)
# -----------------------------------------------------------------------------
@function_tool
def recommend_itinerary(input: ItineraryInput) -> Itinerary:
    """LLM‑only planning guidelines:
    1. If dates are flexible, pick the **nearest future** `departure_date`.
    2. Always choose the **cheapest** flight & hotel satisfying constraints.
    3. If no hotel preference, default to the lowest nightly rate.
    4. Assume 3 nights unless user specifies otherwise and compute `total_cost`.
    5. Offset cost with reward points (1 pt = $0.01) and set `points_used`.
    6. Return **only** JSON that fits the `Itinerary` schema.
    """
    pass

# -----------------------------------------------------------------------------
# In‑memory chat sessions: { user_id: {agent, history (List[str])} }
# -----------------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    logger.info("Request: %s", data)

    # ------------------- extract user payload -------------------
    user_id  = data.get("user_id", "default")
    user_msg = data.get("message", "")
    flights  = data.get("flights", [])
    hotels   = data.get("hotels",  [])
    points   = data.get("user_points", 0)

    # ------------------- initialise session -------------------
    if user_id not in sessions:
        sessions[user_id] = {
            "agent": Agent(
                name="TripBot",
                instructions=(
                    "You are **TripBot**, an expert travel planner.\n"
                    "• Ask only for missing info.\n"
                    "• If user lets you pick dates, choose the nearest future date.\n"
                    "• If user has no hotel preference, pick the cheapest hotel.\n"
                    "• Always choose the cheapest valid flight.\n"
                    "• Once you have flights, hotels and points, call `recommend_itinerary` and reply **only** with JSON."
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini",
            ),
            "history": [],  # list[str]
        }

    agent   = sessions[user_id]["agent"]
    history = sessions[user_id]["history"]  # List[str]

    # ------------------- build conversation string -------------------
    prior_text = "\n".join(history) if history else ""
    details_block = (
        f"FLIGHTS_JSON: {json.dumps(flights)}\n"
        f"HOTELS_JSON:  {json.dumps(hotels)}\n"
        f"USER_POINTS:  {points}"
    )
    conversation = "\n".join(filter(None, [prior_text, user_msg, details_block]))
    logger.info("Conversation sent to agent:\n%s", conversation)

    # ------------------- call the agent -------------------
    try:
        run_result = await Runner.run(agent, conversation)
        logger.info("LLM raw output: %s", run_result.final_output)

        # store turns
        history.extend([user_msg, str(run_result.final_output)])

        # try structured parse
        try:
            itinerary = run_result.final_output_as(Itinerary)
            return JSONResponse(content=itinerary.dict())
        except Exception:
            return JSONResponse(content={"response": str(run_result.final_output)})

    except Exception as e:
        logger.exception("Runner failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
