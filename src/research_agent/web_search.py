import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def web_search(
    query: str,
) -> str:
    """
    Search the web for current information.

    The search is executed by Groq's built-in
    browser search tool.
    """

    response = (
        client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Search the web for reliable information "
                        "relevant to the user's query. "
                        "Return a concise factual summary and "
                        "include the sources provided by the "
                        "browser search."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            tools=[
                {
                    "type": "browser_search",
                }
            ],
            tool_choice="required",
            reasoning_effort="low",
            max_completion_tokens=2048,
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )