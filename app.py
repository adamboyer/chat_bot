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
    """Return ONLY JSON: {"flight_id": "<id>", "hotel_id": "<id>"}."""
    pass

@function_tool
def build_itinerary(input: ItineraryInput) -> Itinerary:
    """Return ONLY JSON matching the Itinerary schema."""
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
    user_msg    = data.get("message", "")
    flights_raw = data.get("flights", [])
    hotels_raw  = data.get("hotels",  [])
    points      = data.get("user_points", 0)

    # ---------------- check for missing data BEFORE any LLM calls ----------------
    missing: List[str] = []
    if not flights_raw:
        missing.append("flights list")
    if not hotels_raw:
        missing.append("hotels list")
    if points == 0:
        missing.append("user_points")

    if missing:
        ask_msg = (
            "I still need the following before I can build your itinerary: "
            + ", ".join(missing)
            + ". Please provide them."
        )
        return JSONResponse({"message": ask_msg, "itinerary": {}})

    # Convert raw → pydantic after we know they exist
    flights = [Flight(**f) for f in flights_raw]
    hotels  = [Hotel(**h) for h in hotels_raw]

    # ---------------- session setup ----------------
    if uid not in sessions:
        sessions[uid] = {
            "selector": Agent(
                name="Selector",
                instructions="""
You are a travel‑planning assistant.

Step 1️⃣  Read the FLIGHTS_JSON and HOTELS_JSON blocks.
Step 2️⃣  Ask the user concise follow‑up questions until you know:
  • preferred departure/arrival cities (or user approves defaults)
  • preferred dates (or confirms flexibility)
  • hotel preference (price vs. quality) or confirms "cheapest"
Step 3️⃣  Once you have enough info, respond with **ONLY** raw JSON that matches the
Selection schema — *no commentary*.  Example:
{"flight_id": "F101", "hotel_id": "HNY1"}
""",
                tools=[choose_options],
                model="gpt-4o-mini",
            ),
            "formatter": Agent(
                name="Formatter",
                instructions="""
You receive:
  • a Selection JSON (flight_id & hotel_id)
  • FLIGHTS_JSON and HOTELS_JSON
  • user_points

1️⃣  Locate the chosen flight & hotel objects.
2️⃣  Calculate total_cost = flight.price + hotel.price_per_night × 3  (assume 3 nights).
3️⃣  Apply user_points at $0.01/pt; set points_used accordingly.
4️⃣  Return **ONLY** valid JSON matching the Itinerary schema — no extra text.
""",
                tools=[build_itinerary],
                model="gpt-4o-mini",
            ),
            "history": []
        }

    sel_agent = sessions[uid]["selector"]
    fmt_agent = sessions[uid]["formatter"]
    history   = sessions[uid]["history"]

    # ------------- selector run -------------
    sel_conv = "\n".join(history + [user_msg]) if history else user_msg
    extra_block = (
        f"FLIGHTS_JSON: {json.dumps(flights_raw)}\n"
        f"HOTELS_JSON:  {json.dumps(hotels_raw)}"
    )
    selector_input = f"{sel_conv}\n{extra_block}"
    logger.info("Selector input→\n%s", selector_input)

    sel_result = await Runner.run(sel_agent, selector_input)
    selector_text = str(sel_result.final_output)
    logger.info("Selector output: %s", selector_text)

    # store conversation text
    history.extend([user_msg, selector_text])

    # ---------------- Parse Selection safely ----------------
    def parse_selection(text: str) -> Optional[Selection]:
        clean = text.strip()
        if clean.startswith("```"):
            clean = "\n".join(
                ln for ln in clean.splitlines()
                if not ln.strip().startswith("```") and not ln.strip().startswith("json")
            )
        try:
            return Selection(**json.loads(clean))
        except Exception:
            return None

    try:
        tmp = sel_result.final_output_as(Selection)
        selection = tmp if isinstance(tmp, Selection) else None
    except Exception:
        selection = None

    if not selection:
        selection = parse_selection(selector_text)

    if not selection:
        logger.warning("Selector output not valid. Asking user to clarify.")
        return JSONResponse({"message": "Could you clarify which flight or hotel you prefer?", "itinerary": {}})

    # ------------- formatter run -------------
    fmt_input = ItineraryInput(
        flights=flights,
        hotels=hotels,
        user_points=points,
        selection=selection,
    )
    fmt_conv = json.dumps(fmt_input.model_dump())
    fmt_result = await Runner.run(fmt_agent, fmt_conv)
    fmt_text = str(fmt_result.final_output)
    logger.info("Formatter output: %s", fmt_text)

    try:
        itinerary = fmt_result.final_output_as(Itinerary).dict()
    except Exception:
        itinerary = {}

    return JSONResponse({"message": fmt_text, "itinerary": itinerary})

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
