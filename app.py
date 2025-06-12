from flask import Flask, request, jsonify
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import os

# Load env
load_dotenv()

# Define output model
class Itinerary(BaseModel):
    flight: dict
    hotel: dict
    total_cost: float
    points_used: int
    notes: str

# Define function tool (agent will use this)
@function_tool
def recommend_itinerary(flights: list, hotels: list, user_points: int) -> Itinerary:
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
        message = data.get("message", "")
        flights = data.get("flights", [])
        hotels = data.get("hotels", [])
        user_points = data.get("user_points", 0)

        inputs = [message]
        if flights or hotels or user_points:
            inputs.append({
                "flights": flights,
                "hotels": hotels,
                "user_points": user_points
            })

        # Run the agent using Runner (sync)
        result = Runner.run(agent, *inputs)

        try:
            # Try to parse structured output
            itinerary = result.final_output_as(Itinerary)
            return jsonify(itinerary.dict())
        except Exception:
            return jsonify({"response": result.output.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
