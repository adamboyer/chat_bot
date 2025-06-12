from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import logging
import os
import uvicorn

# Load env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define input model for function tool
class ItineraryInput(BaseModel):
    flights: list
    hotels: list
    user_points: int

# Define output model
class Itinerary(BaseModel):
    flight: dict
    hotel: dict
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

# Build the agent
agent = Agent(
    name="Itinerary Assistant",
    instructions=(
        "You are a helpful travel planner. Use the `recommend_itinerary` tool when you have enough data "
        "(flights, hotels, and user points). If something is missing, ask the user. Respond clearly and helpfully."
    ),
    tools=[recommend_itinerary],
    model="gpt-4o-mini"
)

# FastAPI setup
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        logger.info("Received request: %s", data)

        message = data.get("message", "")
        flights = data.get("flights", [])
        hotels = data.get("hotels", [])
        user_points = data.get("user_points", 0)

        tool_input = ItineraryInput(
            flights=flights,
            hotels=hotels,
            user_points=user_points
        )

        logger.info("Running agent with input: %s", tool_input)
        result = await Runner.run(agent, message)

        try:
            itinerary = result.final_output_as(Itinerary)
            logger.info("Agent returned structured itinerary: %s", itinerary)
            return JSONResponse(content=itinerary.dict())
        except Exception as parse_err:
            logger.warning("Failed to parse structured output: %s", parse_err)
            return JSONResponse(content={"response": result.output.content})

    except Exception as e:
        logger.error("Error handling request: %s", e, exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
