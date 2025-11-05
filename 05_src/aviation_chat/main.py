from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from aviation_chat.prompts import return_instructions_root
import json
import requests
from utils.logger import get_logger
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")

@tool
def get_cat_facts(n:int=1):
    """
    Returns n cat facts from the Meowfacts API.
    """
    url = "https://meowfacts.herokuapp.com/"
    params = {
        "count": n
    }
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)
    facts_list = resp_dict.get("data", [])
    facts = "\n".join([f"{i+1}. {fact}\n" for i, fact in enumerate(facts_list)])
    return facts

@tool
def get_dog_facts(n:int=1):
    """
    Returns n dog facts from the Dog API.
    """
    url = "http://dogapi.dog/api/v2/facts"
    params = {
        "limit": n
    }
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)
    facts_list = resp_dict.get("data", [])
    facts = "\n".join([f"{i+1}. {fact['attributes']['body']}\n" for i, fact in enumerate(facts_list)])
    return facts

def get_model_with_tools():
    model = init_chat_model(
        "openai:gpt-4o-mini",
        temperature=0.7
    )
    # Augment the LLM with tools
    tools = [get_cat_facts, get_dog_facts]
    model_with_tools = model.bind_tools(tools)
    return model_with_tools

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def semantic_search(query, top_n = 2):
    chroma_client = chromadb.PersistentClient(path="../05_src/documents")
    collection = chroma_client.get_collection(name = "aviation_questions",
                                             embedding_function=OpenAIEmbeddingFunction(
                                                api_key=os.getenv("OPENAI_API_KEY"),
                                                model_name="text-embedding-3-small"
                                            )
                                            )
    results = collection.query(query_texts = query, n_results = top_n)
    return [(id, score, text) for id, score, text in zip(results['ids'][0], results['distances'][0], results['documents'][0])]

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    model_with_tools = get_model_with_tools()

    user_message = state["messages"][-1].content
    faq_answer = semantic_search(user_message)

    _logs.debug(f"faqs: {faq_answer}")

    _logs.debug([
                    SystemMessage(
                        content="You are a helpful assistant designed to provide answers to general flight-related inquiries."
                    )
                ]
                + state["messages"])

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant designed to provide answers to general flight-related inquiries."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""
    tools = [get_cat_facts, get_dog_facts]
    tools_by_name = {tool.name: tool for tool in tools}

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def get_aviation_chat_agent():
    """Returns the aviation chat agent"""  
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()