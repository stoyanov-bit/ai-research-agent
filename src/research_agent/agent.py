import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import time

import logging

from openai import BadRequestError

from research_agent.tools import (
    calculator,
    analyze_csv,
    inspect_csv,
)
from research_agent.retrieval import (
    search_documents,
)

from research_agent.web_search import (
    web_search,
)

from research_agent.models import (
    AgentResult,
    AgentTrace,
    ToolExecution,
)

logger = logging.getLogger(
    __name__
)

load_dotenv()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


SYSTEM_PROMPT = """
You are a research and data analysis agent.

You have access to tools for:
- arithmetic calculations
- CSV inspection
- CSV analysis
- searching local research documents
- searching the public web


=========================================================
TOOL SELECTION
=========================================================

1. Use calculator for ALL arithmetic calculations.

   This also applies when numbers come from another tool.

   Example:
   If analyze_csv returns a mean of 4.5 and the user asks
   for the mean multiplied by 2, first use analyze_csv and
   then use calculator.

   Never perform arithmetic yourself when calculator
   can perform it.


2. Use analyze_csv directly when the user already provides
   the name of the column and asks for a supported statistic.

   Do NOT call inspect_csv first just to verify that the
   column exists.

   analyze_csv already handles column matching and errors.


3. Use inspect_csv only when:
   - the user explicitly asks for the CSV structure,
   - the columns are unknown,
   - the data types are needed,
   - or a previous CSV operation failed because the
     structure was unclear.


4. Use search_documents when the user asks about information
   contained in the available local documents.


5. If the user explicitly names a local document, pass that
   filename as the source argument to search_documents.


6. Use web_search for:
   - current information,
   - recent developments,
   - public web information,
   - or information not available in local documents.


7. Prefer local documents when the user explicitly asks
   about the available documents.


8. Multiple different tools may be used when necessary.


9. Avoid repeating the same tool call unless:
   - the previous call failed,
   - the previous result was insufficient,
   - or a meaningfully different query is necessary.


=========================================================
LOCAL DOCUMENT GROUNDING
=========================================================

10. Claims based on local documents must be supported by
    retrieved text.


11. Every final answer based on search_documents MUST contain
    at least one citation.


12. For text files cite:

    [Source: filename, Chunk: number]


13. For PDF files cite:

    [Source: filename, Page: number, Chunk: number]


14. Use the exact source, page and chunk metadata returned
    by search_documents.


15. Never invent source names, page numbers or chunk numbers.


16. If no sufficiently relevant information is found,
    clearly state that the available documents do not
    contain enough information.


=========================================================
WEB INFORMATION
=========================================================

17. When web search is appropriate, do not present current
    information purely from model memory.


18. Preserve source information returned by web_search
    whenever possible.


=========================================================
GENERAL
=========================================================

19. Tool results are the primary basis for the final answer.


20. If a tool fails, use the error information to determine
    whether another tool call can solve the problem.
"""


TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": (
            "Perform a basic arithmetic operation "
            "on two numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                },
                "b": {
                    "type": "number",
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                    ],
                },
            },
            "required": [
                "a",
                "b",
                "operation",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the public web for current or external information. "
            "Use this tool when the user asks about recent events, "
            "current information, or information that is not contained "
            "in the local documents or CSV files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The question or search query."
                    ),
                },
            },
            "required": [
                "query",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "analyze_csv",
        "description": (
            "Analyze a CSV file. Use this tool when the user asks "
            "for statistics or information contained in a CSV file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the CSV file."
                    ),
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "summary",
                        "mean",
                        "median",
                        "min",
                        "max",
                    ],
                },
                "column": {
                    "type": [
                        "string",
                        "null",
                    ],
                    "description": (
                        "Column to analyze. "
                        "Use null for summary."
                    ),
                },
            },
            "required": [
                "file_path",
                "operation",
                "column",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_csv",
        "description": (
            "Inspect the structure of a CSV file. "
            "Use this tool when you need to know the available "
            "columns, number of rows, or data types before "
            "analyzing the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the CSV file."
                    ),
                },
            },
            "required": [
                "file_path",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_documents",
        "description": (
            "Search local research documents for information. "
            "Use source when the user explicitly names a specific "
            "document. Otherwise search across all documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The information to search for."
                    ),
                },
                "source": {
                    "type": [
                        "string",
                        "null",
                    ],
                    "description": (
                        "Optional exact filename to restrict "
                        "the search to one document. "
                        "Use null to search all documents."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of relevant chunks to return."
                    ),
                },
            },
            "required": [
                "query",
                "source",
                "top_k",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


TOOL_REGISTRY = {
    "calculator": calculator,
    "analyze_csv": analyze_csv,
    "inspect_csv": inspect_csv,
    "search_documents": search_documents,
    "web_search": web_search,
}

def create_agent_response(
    messages,
    max_retries: int = 3,
):
    """
    Call the LLM with retry handling for malformed
    model-generated tool calls.

    Groq may occasionally reject a generated tool call
    when its arguments are not valid JSON.
    """

    last_error = None

    for attempt in range(max_retries):

        try:

            return client.responses.create(
                model="openai/gpt-oss-20b",
                input=messages,
                tools=TOOLS,
            )

        except BadRequestError as error:

            last_error = error

            error_text = str(error)

            is_tool_error = (
                "tool_use_failed"
                in error_text
                or
                "Failed to parse tool call arguments"
                in error_text
            )

            if not is_tool_error:
                raise

            print(
                f"Tool-call generation failed "
                f"(attempt {attempt + 1}/{max_retries}). "
                f"Retrying..."
            )

            if (
                attempt
                < max_retries - 1
            ):
                time.sleep(0.5)

    raise RuntimeError(
        "The model repeatedly generated "
        "an invalid tool call."
    ) from last_error

def has_document_citation(
    answer: str,
) -> bool:
    """
    Check whether the answer contains at least
    one local-document citation.
    """

    return (
        "[Source:"
        in answer
        and
        "Chunk:"
        in answer
    )

def run_agent(
    user_input: str,
    return_trace: bool = False,
):
    """
    Run the research agent.

    The agent repeatedly:
    1. sends the current conversation state to the LLM,
    2. receives tool calls or a final answer,
    3. executes requested tools,
    4. returns tool results to the LLM,
    5. validates document citations,
    6. stops when a valid final answer is produced.

    Parameters
    ----------
    user_input:
        The user's question or task.

    return_trace:
        If True, return an AgentResult containing
        the final answer and the complete execution trace.

        If False, return only the final answer.

    Returns
    -------
    str | AgentResult
        Final answer or structured result with trace.
    """

    # =====================================================
    # Initial conversation state
    # =====================================================

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    # Safety limit to prevent infinite agent loops.
    max_steps = 10

    # Stores the names of all tools used.
    tool_trace = []

    # Stores structured information about every
    # executed tool call.
    tool_results = []

    # The agent gets one opportunity to correct
    # a missing document citation.
    citation_correction_attempts = 0

    # =====================================================
    # Agent loop
    # =====================================================

    for step in range(max_steps):

        # -------------------------------------------------
        # Ask LLM what to do next
        # -------------------------------------------------

        response = create_agent_response(
            messages
        )

        # Store the model output in the conversation state.
        #
        # This is important because Groq does not use
        # previous_response_id in our implementation.
        messages.extend(
            response.output
        )

        # -------------------------------------------------
        # Find tool calls
        # -------------------------------------------------

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        # =================================================
        # No tool call -> model wants to give final answer
        # =================================================

        if not tool_calls:

            answer = (
                response.output_text
            )

            # ---------------------------------------------
            # Citation validation
            # ---------------------------------------------

            used_document_search = (
                "search_documents"
                in tool_trace
            )

            missing_citation = (
                used_document_search
                and not has_document_citation(
                    answer
                )
            )

            # If document retrieval was used but the model
            # forgot the citation, ask it once to correct
            # the answer.
            if (
                missing_citation
                and citation_correction_attempts < 1
            ):

                citation_correction_attempts += 1

                logger.warning(
                    "Document-based answer contained "
                    "no citation. Requesting correction."
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your answer used information from "
                            "search_documents but did not contain "
                            "the required citation. "
                            "Rewrite the answer using the exact "
                            "Source, Page (if available), and Chunk "
                            "metadata returned by the document search. "
                            "Do not invent citations."
                        ),
                    }
                )

                # Return to the beginning of the loop.
                continue

            # ---------------------------------------------
            # Valid final answer
            # ---------------------------------------------

            if return_trace:

                trace = AgentTrace(
                    tools=tool_trace,
                    tool_results=tool_results,
                    steps=step + 1,
                )

                return AgentResult(
                    answer=answer,
                    trace=trace,
                )

            return answer

        # =================================================
        # Execute requested tools
        # =================================================

        for tool_call in tool_calls:

            tool_name = (
                tool_call.name
            )

            arguments = json.loads(
                tool_call.arguments
            )

            logger.info(
                "Tool requested: %s | arguments=%s",
                tool_name,
                arguments,
            )

            tool_trace.append(
                tool_name
            )

            # ---------------------------------------------
            # Find Python implementation
            # ---------------------------------------------

            tool_function = (
                TOOL_REGISTRY.get(
                    tool_name
                )
            )

            success = True
            error_message = None

            # ---------------------------------------------
            # Unknown tool
            # ---------------------------------------------

            if tool_function is None:

                success = False

                error_message = (
                    f"Unknown tool: "
                    f"{tool_name}"
                )

                result = (
                    error_message
                )

                logger.error(
                    "Unknown tool requested: %s",
                    tool_name,
                )

            # ---------------------------------------------
            # Execute known tool
            # ---------------------------------------------

            else:

                try:

                    result = (
                        tool_function(
                            **arguments
                        )
                    )

                    logger.info(
                        "Tool completed: %s",
                        tool_name,
                    )

                except Exception as error:

                    success = False

                    error_message = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    result = (
                        "Tool execution failed: "
                        f"{error_message}"
                    )

                    logger.error(
                        "Tool failed: %s | %s",
                        tool_name,
                        error_message,
                    )

            # ---------------------------------------------
            # Store structured tool execution
            # ---------------------------------------------

            execution = ToolExecution(
                tool=tool_name,
                arguments=arguments,
                result=str(result),
                success=success,
                error=error_message,
            )

            tool_results.append(
                execution
            )

            # ---------------------------------------------
            # Return tool result to LLM
            # ---------------------------------------------

            messages.append(
                {
                    "type": (
                        "function_call_output"
                    ),
                    "call_id": (
                        tool_call.call_id
                    ),
                    "output": str(
                        result
                    ),
                }
            )

    # =====================================================
    # Maximum steps reached
    # =====================================================

    stop_message = (
        "Agent stopped: maximum "
        "number of steps reached."
    )

    logger.warning(
        "Agent reached maximum number "
        "of steps: %s",
        max_steps,
    )

    if return_trace:

        trace = AgentTrace(
            tools=tool_trace,
            tool_results=tool_results,
            steps=max_steps,
        )

        return AgentResult(
            answer=stop_message,
            trace=trace,
        )

    return stop_message