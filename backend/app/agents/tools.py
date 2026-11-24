import json
from datetime import datetime

import httpx
from langchain_core.documents import Document
from langchain_core.tools import tool

from app.rag.retriever import search


@tool(response_format="content_and_artifact")
async def search_documents(query: str) -> tuple[str, list[Document]]:
    """Search uploaded documents for information relevant to a question."""
    documents = await search(query)
    content = json.dumps([
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page"),
         "text": doc.page_content}
        for doc in documents
    ], ensure_ascii=False)
    return content, documents


@tool
async def current_time() -> str:
    """Get the current server date and time with timezone."""
    return datetime.now().astimezone().isoformat()


@tool
async def get_weather(city: str) -> str:
    """Get current temperature and wind for a city using Open-Meteo."""
    city = city.strip()
    if not city or len(city) > 100:
        raise ValueError("City must contain 1 to 100 characters")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        response.raise_for_status()
        locations = response.json().get("results", [])
        if not locations:
            raise ValueError(f"No location found for {city}")
        location = locations[0]
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,wind_speed_10m",
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        forecast = response.json()
    return json.dumps({
        "location": location["name"],
        "country": location.get("country"),
        "timezone": forecast["timezone"],
        "current": forecast["current"],
        "units": forecast["current_units"],
    })