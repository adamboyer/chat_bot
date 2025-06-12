from flask import Flask, request, jsonify
from openai import OpenAI, AssistantRunnable, ChatSession
from openai.tools import function_tool
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool: Only a function signature and description
@function_tool
def recommend_itinerary(flights: list, hotels: list, user_points: int) -> dict:
    """
    Recommend a travel itinerary using available flights, hotels, and user points.
    Wait until you have all 3 inputs. Ask the user for missing info if needed.
    Return a JSON object with fields: flight, hotel, total_cost, points_used, and notes.
    """
    pass

# Agent with planning instructions
agent = AssistantRunnable.from_tools(
    tools=[recommend_itinerary],
    instructions="""
    You are a travel planning assistant. Collect any missing information from the user
    (flights, hotels, or user points). Once you have everything, use the tool to create a complete itinerary.
    Always keep track of what the user has already told you. Speak clearly and respond in a helpful tone.
    """
)

# Flask app setup
app = Flask(__name__)

# Track simple in-memory session state (use session ID/IP for real apps)
session_map = {}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_id = request.remote_addr  # better: use a real session ID
        data = request.get_json()
        message = data.get("message")

        if user_id not in session_map:
            session_map[user_id] = ChatSession(agent)

        session = session_map[user_id]
        result = session.invoke(message)
        return jsonify({"response": result.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Dev server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
