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
    departure_date: Optional[str] = None
    arrival_date:   Optional[str] = None
    price: float

class Hotel(BaseModel):
    id: str
    name: str
    price_per_night: float

class Selection(BaseModel):
    flight_id: str
    hotel_id: str

class ItineraryInput(BaseModel):
    flights: List[Flight]
    hotels: List[Hotel]
    user_points: int
    selection: Selection

class Itinerary(BaseModel):
    flight: Flight
    hotel: Hotel
    total_cost: float
    points_used: int
    notes: str

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
@function_tool
def choose_options(flights: List[Flight], hotels: List[Hotel]) -> Selection:
    """LLM task: from the lists, pick the cheapest flight and cheapest hotel (or follow user hints) and return their IDs."""
    pass

@function_tool
def build_itinerary(input: ItineraryInput) -> Itinerary:
    """LLM task: using the chosen flight & hotel IDs plus reward points, compute total_cost, points_used, and return a full itinerary JSON."""
    pass

# -----------------------------------------------------------------------------
# In‑memory sessions
# -----------------------------------------------------------------------------
app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    logger.info("Request: %s", data)

    uid  = data.get("user_id", "default")
    msg  = data.get("message", "")
    flights_json = data.get("flights", [])
    hotels_json  = data.get("hotels",  [])
    points       = data.get("user_points", 0)

    flights = [Flight(**f) for f in flights_json]
    hotels  = [Hotel(**h) for h in hotels_json]

    if uid not in sessions:
        # Planner agent – picks best IDs
        selector = Agent(
            name="Selector",
            instructions="Choose the best flight & hotel IDs per user criteria or default to cheapest/nearest.",
            tools=[choose_options],
            model="gpt-4o-mini"
        )
        # Formatter agent – returns full JSON itinerary
        formatter = Agent(
            name="Formatter",
            instructions="Given a chosen flight & hotel and points, call build_itinerary and return only JSON.",
            tools=[build_itinerary],
            model="gpt-4o-mini"
        )
        sessions[uid] = {"selector": selector, "formatter": formatter, "history": []}

    sel_agent = sessions[uid]["selector"]
    fmt_agent = sessions[uid]["formatter"]
    history   = sessions[uid]["history"]

    # ---------- First agent: pick IDs ----------
    sel_conv = "\n".join(history + [msg]) if history else msg
    logger.info("Selector conversation:\n%s", sel_conv)
    sel_result = await Runner.run(sel_agent, sel_conv, flights_json, hotels_json)
    logger.info("Selector output: %s", sel_result.final_output)
    history.extend([msg, str(sel_result.final_output)])

    try:
        selection = sel_result.final_output_as(Selection)
    except Exception:
        return JSONResponse(content={"message": str(sel_result.final_output), "itinerary": {}})

    # ---------- Second agent: build itinerary ----------
    fmt_input = ItineraryInput(flights=flights, hotels=hotels, user_points=points, selection=selection)
    fmt_conv  = json.dumps(fmt_input.model_dump())
    fmt_result = await Runner.run(fmt_agent, fmt_conv)
    logger.info("Formatter output: %s", fmt_result.final_output)

    try:
        itinerary = fmt_result.final_output_as(Itinerary).dict()
    except Exception:
        itinerary = {}

    return JSONResponse(content={
        "message": str(fmt_result.final_output),
        "itinerary": itinerary
    })

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
