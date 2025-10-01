import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


#  Set API Keys

os.environ["GOOGLE_API_KEY"] = "Google_API_Key"
os.environ["TAVILY_API_KEY"] = "Tavily_API_Key"
OPENWEATHER_API_KEY = "your-openweather-api-key" 


#  Define Weather Tool

@tool
def get_weather(city: str) -> str:
    """Fetch current weather for a city using OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url).json()
    if res.get("weather"):
        return f"Weather in {res['name']}: {res['main']['temp']}°C, {res['weather'][0]['description']}"
    return "Could not fetch weather."


#  Create the Agent

memory = MemorySaver()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

search = TavilySearch(max_results=2)
tools = [search, get_weather]  # include weather + search

agent_executor = create_react_agent(model, tools, checkpointer=memory)


#  Conversation

config = {"configurable": {"thread_id": "abc123"}}

# First message (agent should remember)
input_message = {
    "role": "user",
    "content": "Hi, I'm xylo and I live in gurgaon,india"
    "."
}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

# Second message (agent should recall SF + fetch weather)
input_message = {
    "role": "user",
    "content": "What's the weather where I live?"
}
for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()
