from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from openai.agents import Runnable, ChatSession
from openai.tools import function_tool
from pydantic import BaseModel
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a Pydantic model for structured return
class Itinerary(BaseModel):
    flight: dict
    hotel: dict
    total_cost: float
    points_used: int
    notes: str

# Agent-accessible tool
@function_tool
def recommend_itinerary(flights: list, hotels: list, user_points: int) -> Itinerary:
    """Use this tool to recommend the best itinerary using flights, hotels, and reward points."""
    pass  # OpenAI will reason and respond

# Create the agent with instructions
agent = Runnable.tools([recommend_itinerary]).with_instructions("""
You are a travel planning assistant. Gather all necessary info from the user
(flights, hotels, user_points). Once all data is available, use the recommend_itinerary tool
to create a full travel plan. Respond in a helpful, friendly tone.
""")

# Simple in-memory session store
user_sessions = {}

# Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_id = request.remote_addr  # for real use, replace with session or token
        data = request.json
        message = data.get("message")

        if not message:
            return jsonify({"error": "Missing 'message' field"}), 400

        # Reuse ChatSession or start new
        if user_id not in user_sessions:
            user_sessions[user_id] = ChatSession(agent)

        session = user_sessions[user_id]

        # Invoke the agent with the user's message
        reply = session.invoke(message)

        return jsonify({"response": reply.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
