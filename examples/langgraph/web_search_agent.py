import os

import dotenv
from langchain_community.tools import (DuckDuckGoSearchResults,
                                       DuckDuckGoSearchRun)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
search_tool = DuckDuckGoSearchResults(api_wrapper=search_wrapper)

agent = create_react_agent(
    model=model,
    tools=[search_tool],
    prompt=(
        "You are a professional financial news reporter "
        "with expertise in analyzing and summarizing financial news articles. "
        "Your task is to search for the latest financial news, "
        "analyze the content, and provide a concise summary of the key points."
    ),
)

if __name__ == "__main__":
    for step in agent.stream(
        {
            "messages": [
                ("human", "What is the latest status of the Vietnam stock market?")
            ]
        },
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
