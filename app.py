from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import os

# Load .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define output model
class Itinerary(BaseModel):
    flight: dict
    hotel: dict
    total_cost: float
    points_used: int
    notes: str

# Define tool for agent
@function_tool
def recommend_itinerary(flights: list, hotels: list, user_points: int) -> Itinerary:
    """
    Use this tool to select the best flight and hotel based on price and points.
    Return the complete itinerary in JSON format.
    """
    pass  # LLM will handle logic

# Create agent
agent = Agent.tools([recommend_itinerary]).with_instructions("""
You are a travel planner. Collect flights, hotels, and user points from the user,
and when you have all three, use the tool to create the best itinerary based on value.
""")

# Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "")
        flights = data.get("flights", [])
        hotels = data.get("hotels", [])
        user_points = data.get("user_points", 0)

        # Compose input message list for the agent
        inputs = [message]

        # If structured data exists, pass as input to tool
        if flights or hotels or user_points:
            inputs.append({
                "flights": flights,
                "hotels": hotels,
                "user_points": user_points
            })

        # Run the agent using Runner (synchronous version)
        result = Runner.run(agent, *inputs)

        # If tool was called, extract structured output
        try:
            itinerary = result.final_output_as(Itinerary)
            return jsonify(itinerary.dict())
        except Exception:
            # No tool used, fallback to LLM message
            return jsonify({"response": result.output.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
