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
def get_weather(city: str):
    """
    Retrieves and formats current weather information for the given city.
    Example: get_weather("Toronto")
    """
    import os, requests, json

    url = "https://api.weatherstack.com/current"
    params = {
        "access_key": os.getenv("WEATHER_API_KEY"),
        "query": city
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return format_weather_info(data)

    except requests.exceptions.RequestException as e:
        return f"⚠️ Unable to retrieve weather data for {city}. ({str(e)})"
    except Exception as e:
        return f"⚠️ An error occurred while processing the weather data. ({str(e)})"


def format_weather_info(api_response):
    """
    Formats the weatherstack API response into a human-readable string.
    """
    if "current" not in api_response or "location" not in api_response:
        return "⚠️ Weather information is unavailable."

    location = api_response["location"]
    current = api_response["current"]

    city_name = location.get("name")
    country = location.get("country")
    temperature = current.get("temperature")
    feelslike = current.get("feelslike")
    description = ", ".join(current.get("weather_descriptions", []))
    humidity = current.get("humidity")
    wind_speed = current.get("wind_speed")
    wind_dir = current.get("wind_dir")
    localtime = location.get("localtime")

    formatted = (
        f"🌍 **Weather Report for {city_name}, {country}**\n"
        f"🕒 Local Time: {localtime}\n"
        f"🌡️ Temperature: {temperature}°C (Feels like {feelslike}°C)\n"
        f"🌤️ Condition: {description}\n"
        f"💧 Humidity: {humidity}%\n"
        f"🌬️ Wind: {wind_speed} km/h {wind_dir}\n"
    )

    return formatted

@tool
def get_flight_status(flight_num:str):
    """
    Returns flight information for flight_num.
    """
    url = f"https://api.aviationstack.com/v1/flights"
    params = {
        "access_key": os.getenv("AVIATION_API_KEY"),
        "flight_iata": flight_num
    }
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)
    return format_flight_info(resp_dict)

def format_flight_info(api_response):
    formatted = []
    for flight in api_response.get("data", []):
        flight_info = {
            "flight": {
                "airline": flight["airline"]["name"],
                "flight_number": flight["flight"]["iata"],
                "status": flight["flight_status"].capitalize(),
                "date": flight["flight_date"]
            },
            "departure": {
                "airport": flight["departure"]["airport"],
                "iata": flight["departure"]["iata"],
                "terminal": flight["departure"]["terminal"],
                "gate": flight["departure"]["gate"],
                "scheduled_time": flight["departure"]["scheduled"],
                "delay_minutes": flight["departure"]["delay"]
            },
            "arrival": {
                "airport": flight["arrival"]["airport"],
                "iata": flight["arrival"]["iata"],
                "terminal": flight["arrival"]["terminal"],
                "baggage_claim": flight["arrival"]["baggage"],
                "scheduled_time": flight["arrival"]["scheduled"]
            }
        }
        formatted.append(flight_info)
    return str(formatted)

def semantic_search(query, top_n:int=2, jsonl_path="../05_src/aviation_chat/faq_data.jsonl"):
    """
    Performs semantic search and returns the answers of the top N results.
    """
    results = query_chromadb(query, top_n=top_n)

    if not results or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    scores = results["distances"][0]
    questions = results["documents"][0]

    # Get answers in one pass
    answers_dict = get_answers_by_ids(jsonl_path, ids)

    # Combine results
    combined_results = []
    for i, id in enumerate(ids):
        combined_results.append({
            "id": id,
            "score": scores[i],
            "question": questions[i],
            "answer": answers_dict.get(id)
        })

    return combined_results

def query_chromadb(query, top_n:int=2):
    """
    Please use this first everytime. Return n queries in database that matches the prompt.
    """
    chroma_client = chromadb.PersistentClient(path="../05_src/documents")
    collection = chroma_client.get_collection(
        name="aviation_questions",
        embedding_function=OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
    )

    results = collection.query(query_texts=[query], n_results=top_n)
    
    formatted_results = []
    for id, score, text in zip(results['ids'][0], results['distances'][0], results['documents'][0]):
        formatted_results.append(f"[{id}] (Score: {score:.4f}) → {text}")

    return results


def get_answers_by_ids(jsonl_path, target_ids):
    """
    Reads a JSONL file and returns answers for multiple IDs.
    Returns a dict {id: answer}.
    """
    target_ids = set(target_ids)  # for faster lookup
    answers = {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            record_id = record.get("id")
            if record_id in target_ids:
                answers[record_id] = record.get("answer")

            # Optional: break early if all IDs found
            if len(answers) == len(target_ids):
                break

    return answers

def get_model_with_tools():
    model = init_chat_model(
        "openai:gpt-4o-mini",
        temperature=0.7
    )
    # Augment the LLM with tools
    tools = [get_flight_status, get_weather]
    model_with_tools = model.bind_tools(tools)
    return model_with_tools

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    model_with_tools = get_model_with_tools()

    user_message = state["messages"][-1].content
    faq = semantic_search(user_message)

    _logs.info(faq)

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=return_instructions_root()+
                        "If the answer matches the context, give the result given in the following:"
                        f"\n\n{faq}"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""
    tools = [get_flight_status, get_weather]
    tools_by_name = {tool.name: tool for tool in tools}

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        _logs.debug(f"observation: {observation}")
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