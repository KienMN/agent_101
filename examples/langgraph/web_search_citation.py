import dotenv
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field

dotenv.load_dotenv()

system_prompt = (
    "You're a helpful AI assistant. Given a user question "
    "and some website snippets, answer the user "
    "question. If none of the websites answer the question, "
    "just say you don't know."
    "\n\nHere are the website searched: "
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
# prompt.pretty_print()
model = ChatGroq(model="llama-3.3-70b-versatile")


class Citation(BaseModel):
    text: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )
    url: str = Field(..., description="The URL of the source")


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the question, which is based only on the given sources.",
    )
    citation: list[Citation] = Field(
        ..., description="The citation that justifies the answer."
    )


class State(BaseModel):
    question: str
    context: list[dict] | None = None
    answer: QuotedAnswer | None = None


def format_docs_with_id(docs: list[dict]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc["title"]}\nArticle Snippet: {doc["snippet"]}\nURL: {doc["link"]}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


def retrieve(
    state: State,
) -> State:
    """Retrieve relevant documents from the web."""
    search_tool = DuckDuckGoSearchResults(
        api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3),
        output_format="list",
    )
    search_run = DuckDuckGoSearchRun(tool=search_tool)
    results = search_tool.invoke(state.question)
    # print(results)
    state.context = results
    return state


def generate_answer(
    state: State,
) -> State:
    """Generate an answer based on the retrieved documents."""
    formatted_docs = format_docs_with_id(state.context)
    messages = prompt.invoke({"question": state.question, "context": formatted_docs})

    structured_llm = model.with_structured_output(QuotedAnswer)
    response = structured_llm.invoke(messages)
    state.answer = response
    return state


graph_builder = StateGraph(State).add_sequence([retrieve, generate_answer])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "How fast are cheetahs?"})
print(result["answer"])
