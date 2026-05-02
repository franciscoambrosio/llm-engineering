# -*- coding: utf-8 -*-
"""
Module 04 — Agents
==================

Problem: LLMs are great at reasoning, but can't take actions.

Solution: Give the LLM tools and let it decide when to use them.

This module covers:
  1. Tool definitions: what tools to give the LLM
  2. Tool use: LLM calling tools
  3. Agent loop: repeat until done
  4. Control: system prompts and max iterations

Run:
  python 04_agents/agents.py
"""

import os
import json
import time
from datetime import datetime
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone"}
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform math calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_weather(city: str) -> dict:
    weather_data = {
        "Paris": {"temp": 15, "condition": "rainy"},
        "Tokyo": {"temp": 18, "condition": "cloudy"},
        "London": {"temp": 12, "condition": "rainy"},
    }
    return weather_data.get(city, {"error": "City not found"})


def get_time(timezone: str) -> dict:
    return {
        "timezone": timezone,
        "time": datetime.now().isoformat()
    }


def calculate(expression: str) -> dict:
    try:
        import math
        result = eval(expression, {"sqrt": math.sqrt, "__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def execute_tool(name: str, args: dict) -> dict:
    tools = {
        "get_weather": get_weather,
        "get_time": get_time,
        "calculate": calculate,
    }
    if name not in tools:
        return {"error": f"Unknown tool: {name}"}
    return tools[name](**args)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """Agent with tool use, system prompts, and iteration control."""

    def __init__(self, max_iterations: int = 5, system_prompt: Optional[str] = None):
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or "You are a helpful assistant with access to tools."
        self.messages = []
        self.iteration = 0

    def call_groq(self, messages: list) -> dict:
        """Call Groq API with tool use."""
        import requests

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",  # Free and fast
                "max_tokens": 1024,
                "temperature": 0.3,
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *messages
                ]
            },
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Groq error: {response.text}")

        return response.json()

    def run(self, user_query: str) -> str:
        """Run agent loop."""
        print(f"\n🤖 Agent starting...")
        print(f"Query: {user_query}")
        print(f"Max iterations: {self.max_iterations}\n")

        self.messages = [{"role": "user", "content": user_query}]

        for iteration in range(self.max_iterations):
            self.iteration = iteration + 1
            iterations_left = self.max_iterations - self.iteration

            if iterations_left <= 1:
                print(f"⚠️  Warning: Only {iterations_left} iteration(s) left")

            print(f"── Step {self.iteration}/{self.max_iterations} ─────────────────────────")

            try:
                response = self.call_groq(self.messages)
            except Exception as e:
                return f"Error: {e}"

            choice = response["choices"][0]
            finish_reason = choice["finish_reason"]
            message = choice["message"]

            # Handle tool calls
            if finish_reason == "tool_calls" and "tool_calls" in message:
                print(f"Assistant: {message.get('content', '')}")
                self.messages.append(message)

                tool_results = []
                for tool_call in message["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])

                    print(f"🔧 {name}({json.dumps(args)})")
                    result = execute_tool(name, args)
                    print(f"   Result: {json.dumps(result)}")

                    # Groq expects tool_call_id not tool_call_id
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": json.dumps(result)
                    })

                # Add tool results as separate user message
                # Format: role:user with tool_results content
                self.messages.append({
                    "role": "user",
                    "content": json.dumps(tool_results)
                })

            else:
                # Final response
                response_text = message.get("content", "No response")
                print(f"\n✓ Complete ({self.iteration} steps)\n")
                print(f"Response: {response_text}")
                return response_text

        return f"Failed: Max iterations ({self.max_iterations}) reached"


# ─────────────────────────────────────────────────────────────────────────────
# 4. TESTS & DEMOS
# ─────────────────────────────────────────────────────────────────────────────

def test_tools():
    print("\n── Tests ────────────────────────────────────────────────────")
    assert get_weather("Paris")["temp"] == 15
    assert calculate("2+2")["result"] == 4
    assert get_time("UTC")["timezone"] == "UTC"
    print("  ✓  All tools work\n")


def demo_live():
    print("── Live Agent Demo ──────────────────────────────────────────")

    agent1 = Agent(max_iterations=5)
    agent1.run("What is the weather in Paris?")

    print("\n" + "─" * 70)

    agent2 = Agent(
        max_iterations=3,
        system_prompt="Be concise. Use tools only if necessary."
    )
    agent2.run("Calculate 2 + 2")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_tools()

    groq_key = os.getenv("GROQ_API_KEY")
    print("─" * 70)
    if groq_key:
        print("GROQ_API_KEY found. Running live demos...\n")
        try:
            demo_live()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("GROQ_API_KEY not set. To run live demos:")
        print("  export GROQ_API_KEY=your_key")
