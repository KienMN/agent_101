from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description=(
        "You are a professional financial news reporter "
        "with expertise in analyzing and summarizing financial news articles. "
        "Your task is to search for the latest financial news, "
        "analyze the content, and provide a concise summary of the key points."
    ),
    tools=[
        DuckDuckGoTools(
            fixed_max_results=1,
        )
    ],
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "What is the latest status of the Vietnam stock market?",
        show_full_reasoning=True,
        stream_intermediate_steps=True,
    )
