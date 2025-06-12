from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging
import os
import uvicorn
from typing import List, Dict

# Load env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define structured item models
class Flight(BaseModel):
    id: str
    departure: str
    arrival: str
    price: float

class Hotel(BaseModel):
    id: str
    name: str
    price_per_night: float

# Define input model for function tool
class ItineraryInput(BaseModel):
    flights: List[Flight]
    hotels: List[Hotel]
    user_points: int

# Define output model
class Itinerary(BaseModel):
    flight: Flight
    hotel: Hotel
    total_cost: float
    points_used: int
    notes: str

# Define function tool (agent will use this)
@function_tool
def recommend_itinerary(input: ItineraryInput) -> Itinerary:
    """
    Choose the best flight and hotel based on price and user points.
    Return a full itinerary in structured JSON.
    """
    pass  # The agent will handle this logic internally

# Store chat sessions keyed by user_id
sessions: Dict[str, Runner] = {}

# FastAPI setup
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        logger.info("Received request: %s", data)

        user_id = data.get("user_id", "default")
        message = data.get("message", "")
        flights = data.get("flights", [])
        hotels = data.get("hotels", [])
        user_points = data.get("user_points", 0)

        tool_input = ItineraryInput(
            flights=[Flight(**f) for f in flights],
            hotels=[Hotel(**h) for h in hotels],
            user_points=user_points
        )

        if user_id not in sessions:
            agent = Agent(
                name="Itinerary Assistant",
                instructions=(
                    "You are a helpful travel planner. Use the `recommend_itinerary` tool when you have enough data "
                    "(flights, hotels, and user points). If something is missing, ask the user. Respond clearly and helpfully."
                ),
                tools=[recommend_itinerary],
                model="gpt-4o-mini"
            )
            sessions[user_id] = await Runner.create_session(agent)

        logger.info("Running agent with input: %s", tool_input)
        result = await sessions[user_id].run(message)

        try:
            itinerary = result.final_output_as(Itinerary)
            logger.info("Agent returned structured itinerary: %s", itinerary)
            return JSONResponse(content=itinerary.dict())
        except Exception as parse_err:
            logger.warning("Failed to parse structured output: %s", parse_err)
            logger.info("Raw result content: %s", str(result))
            return JSONResponse(content={"response": str(result)})

    except Exception as e:
        logger.error("Error handling request: %s", e, exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
