from flask import Flask, request, jsonify
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import os
import asyncio
import logging

# Load env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define output model
class Itinerary(BaseModel):
    flight: dict
    hotel: dict
    total_cost: float
    points_used: int
    notes: str

# Define function tool (agent will use this)
@function_tool
def recommend_itinerary(input: dict) -> Itinerary:
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
    model="gpt-4o-mini"  # You can change this model if needed
)

# Flask setup
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        logger.info("Received request: %s", data)

        message = data.get("message", "")
        flights = data.get("flights", [])
        hotels = data.get("hotels", [])
        user_points = data.get("user_points", 0)

        tool_input = {
            "flights": flights,
            "hotels": hotels,
            "user_points": user_points
        }

        logger.info("Running agent with inputs: [%s, %s]", message, tool_input)
        result = asyncio.run(Runner.run(agent, message, tool_input))

        try:
            itinerary = result.final_output_as(Itinerary)
            logger.info("Agent returned structured itinerary: %s", itinerary)
            return jsonify(itinerary.dict())
        except Exception as parse_err:
            logger.warning("Failed to parse structured output: %s", parse_err)
            return jsonify({"response": result.output.content})

    except Exception as e:
        logger.error("Error handling request: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
