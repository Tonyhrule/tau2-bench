import os
from time import sleep
from codex import Client
import httpx
from pydantic import TypeAdapter
from openai.types.chat import ChatCompletionMessageParam

GuidanceResults = TypeAdapter(list[str])


def _attempt_request(project_id, query, message_history, api_key, base_url, attempts=0):
    try:
        response = httpx.post(
            base_url.join(f"api/projects/{project_id}/consult"),
            json={"query": query, "message_history": message_history},
            headers={"X-API-Key": api_key},
        )
        return response.json()["guidance"]
    except:
        if attempts >= 10:
            raise RuntimeError("Failed to get guidance after 10 attempts")
        sleep(1)
        return _attempt_request(
            project_id, query, message_history, api_key, base_url, attempts + 1
        )


def consult(query: str, message_history: list[ChatCompletionMessageParam]) -> list[str]:
    api_key = os.getenv("CODEX_API_KEY")
    if not api_key:
        raise ValueError("CODEX_API_KEY environment variable is not set")
    client = Client(api_key=api_key)
    base_url = client._client.base_url
    project_id = os.getenv("CLEANLAB_PROJECT_ID")
    if not project_id:
        raise ValueError("CLEANLAB_PROJECT_ID environment variable is not set")
    res = _attempt_request(project_id, query, message_history, api_key, base_url)
    return GuidanceResults.validate_python(res)
