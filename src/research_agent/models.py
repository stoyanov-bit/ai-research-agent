from typing import Any

from pydantic import BaseModel, Field


class ToolExecution(BaseModel):
    """
    Represents one executed tool call.
    """

    tool: str

    arguments: dict[str, Any]

    result: str

    success: bool = True

    error: str | None = None


class AgentTrace(BaseModel):
    """
    Structured trace of an agent run.
    """

    tools: list[str] = Field(
        default_factory=list
    )

    tool_results: list[ToolExecution] = Field(
        default_factory=list
    )

    steps: int = 0


class AgentResult(BaseModel):
    """
    Final result returned by the agent.
    """

    answer: str

    trace: AgentTrace