from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging, json, uvicorn
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
    """LLM task: pick the best flight & hotel IDs (typically the cheapest unless user hints otherwise)."""
    pass

@function_tool
def build_itinerary(input: ItineraryInput) -> Itinerary:
    """LLM task: with chosen IDs + reward points, compute cost / points_used and return JSON itinerary."""
    pass

# -----------------------------------------------------------------------------
# FastAPI + per‑user sessions
# -----------------------------------------------------------------------------
app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    logger.info("Request: %s", data)

    uid         = data.get("user_id", "default")
    msg         = data.get("message", "")
    flights_json = data.get("flights", [])
    hotels_json  = data.get("hotels",  [])
    points       = data.get("user_points", 0)

    flights = [Flight(**f) for f in flights_json]
    hotels  = [Hotel(**h) for h in hotels_json]

    # --------------------------- session setup ---------------------------
    if uid not in sessions:
        sessions[uid] = {
            "selector": Agent(
                name="Selector",
                instructions="Choose best flight & hotel IDs (cheapest or per user hints).",
                tools=[choose_options],
                model="gpt-4o-mini",
            ),
            "formatter": Agent(
                name="Formatter",
                instructions="Given flight+hotel IDs and points, call build_itinerary and output JSON only.",
                tools=[build_itinerary],
                model="gpt-4o-mini",
            ),
            "history": []  # List[str]
        }

    sel_agent = sessions[uid]["selector"]
    fmt_agent = sessions[uid]["formatter"]
    history   = sessions[uid]["history"]

    # --------------------------- selector step ---------------------------
    sel_conv = "\n".join(history + [msg]) if history else msg
    extra_block = (
        f"FLIGHTS_JSON: {json.dumps(flights_json)}\n"
        f"HOTELS_JSON:  {json.dumps(hotels_json)}"
    )
    selector_input = "\n".join([sel_conv, extra_block])
    logger.info("Selector conversation passed to LLM:\n%s", selector_input)

    sel_result = await Runner.run(sel_agent, selector_input)
    logger.info("Selector raw output: %s", sel_result.final_output)

    history.extend([msg, str(sel_result.final_output)])

    try:
        selection = sel_result.final_output_as(Selection)
    except Exception:
        return JSONResponse(content={"message": str(sel_result.final_output), "itinerary": {}})

    # --------------------------- formatter step ---------------------------
    fmt_input = ItineraryInput(
        flights=flights,
        hotels=hotels,
        user_points=points,
        selection=selection,
    )
    fmt_conv = json.dumps(fmt_input.model_dump())
    fmt_result = await Runner.run(fmt_agent, fmt_conv)
    logger.info("Formatter raw output: %s", fmt_result.final_output)

    try:
        itinerary_dict = fmt_result.final_output_as(Itinerary).dict()
    except Exception:
        itinerary_dict = {}

    return JSONResponse(content={
        "message": str(fmt_result.final_output),
        "itinerary": itinerary_dict,
    })

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
