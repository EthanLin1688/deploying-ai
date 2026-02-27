# Aviation Chat Agent

This chat agent can check flight status, look up weather, and answer common questions using OpenAI’s model and LangGraph. It gives quick, real-time answers about flights and weather through connected APIs.

## Services
- Flight Status Retrieval – Fetches live flight details (departure, arrival, and status) using the AviationStack API

- Weather Search – Provides current weather conditions using the WeatherStack API

- Semantic Search – Retrieves relevant answers from a FAQ database using ChromaDB for embedding-based search.
Semantic search is used to ensure the model doesn't hallucinate on topics related to safety or responsibility.

run using python -m aviation_chat.app