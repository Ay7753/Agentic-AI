import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


#  Set API keys

os.environ["GOOGLE_API_KEY"] = "AIzaSyAGcpkY6nbEgfmajuQYkA20FALucck6Jrw"
os.environ["TAVILY_API_KEY"] = "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"


# Initialize memory & model

memory = MemorySaver()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


#  Tools

search = TavilySearch(max_results=3)
tools = [search]


#  Create ReAct agent

agent_executor = create_react_agent(model, tools, checkpointer=memory)


#  Start conversation

config = {"configurable": {"thread_id": "research_thread"}}

def ask_agent(question: str):
    input_message = {
        "role": "user",
        "content": question,
    }
    print("\n===== HUMAN =====")
    print(input_message["content"])
    print("\n===== AGENT =====")
    for step in agent_executor.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()


# Example usage

ask_agent("I want to research recent advancements in Agentic AI.")
ask_agent("Summarize 2-3 recent papers and give references.")
ask_agent("Can you suggest follow-up research questions?")
