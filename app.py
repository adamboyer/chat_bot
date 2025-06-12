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
    """Return JSON:
    {"flight_id": "<id>", "hotel_id": "<id>"}
    Pick the cheapest flight & hotel (unless user hints otherwise). Output **only** the JSON.
    """
    pass

@function_tool
def build_itinerary(input: ItineraryInput) -> Itinerary:
    """Given chosen IDs + points, compute cost, points_used, and return an Itinerary JSON."""
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

    uid   = data.get("user_id", "default")
    msg   = data.get("message", "")
    flights_raw = data.get("flights", [])
    hotels_raw  = data.get("hotels",  [])
    points      = data.get("user_points", 0)

    flights = [Flight(**f) for f in flights_raw]
    hotels  = [Hotel(**h) for h in hotels_raw]

    # ---------------- session setup ----------------
    if uid not in sessions:
        sessions[uid] = {
            "selector": Agent(
                name="Selector",
                instructions=(
                    "Pick best flight & hotel IDs (cheapest/default).\n"
                    "Respond with JSON matching Selection schema only."
                ),
                tools=[choose_options],
                model="gpt-4o-mini",
            ),
            "formatter": Agent(
                name="Formatter",
                instructions="Use build_itinerary to return JSON itinerary only.",
                tools=[build_itinerary],
                model="gpt-4o-mini",
            ),
            "history": []
        }

    sel_agent = sessions[uid]["selector"]
    fmt_agent = sessions[uid]["formatter"]
    history   = sessions[uid]["history"]

    # ------------- selector run -------------
    sel_conv = "\n".join(history + [msg]) if history else msg
    extra_block = (
        f"FLIGHTS_JSON: {json.dumps(flights_raw)}\n"
        f"HOTELS_JSON:  {json.dumps(hotels_raw)}"
    )
    selector_input = "\n".join([sel_conv, extra_block])
    logger.info("Selector input→\n%s", selector_input)

    sel_result = await Runner.run(sel_agent, selector_input)
    selector_text = str(sel_result.final_output)
    logger.info("Selector output: %s", selector_text)

    history.extend([msg, selector_text])

    # ---------------- Parse Selection safely ----------------
    def _parse_selection(text: str) -> Optional[Selection]:
        """Try multiple strategies to coerce raw LLM text → Selection"""
        raw = text.strip()
        # remove markdown fences
        if raw.startswith("```"):
            raw = "".join(
                line for line in raw.splitlines()
                if not line.strip().startswith("```") and not line.strip().startswith("json")
            )
        try:
            data = json.loads(raw)
            return Selection(**data)
        except Exception:
            return None

    selection = sel_result.final_output_as(Selection, default=None)
    if not isinstance(selection, Selection):
        selection = _parse_selection(selector_text)

    if not selection:
        logger.warning("Selector output still not valid after cleaning.")
        return JSONResponse({"message": selector_text, "itinerary": {}})

    # ------------- formatter run -------------
    fmt_input = ItineraryInput(flights=flights, hotels=hotels, user_points=points, selection=selection)
    fmt_conv = json.dumps(fmt_input.model_dump())
    fmt_result = await Runner.run(fmt_agent, fmt_conv)
    fmt_text = str(fmt_result.final_output)
    logger.info("Formatter output: %s", fmt_text)

    try:
        itinerary = fmt_result.final_output_as(Itinerary).dict()
    except Exception:
        itinerary = {}

    return JSONResponse({
        "message": fmt_text,
        "itinerary": itinerary
    })

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
