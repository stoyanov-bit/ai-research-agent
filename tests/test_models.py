from research_agent.models import (
    AgentResult,
    AgentTrace,
    ToolExecution,
)


def test_tool_execution():

    execution = ToolExecution(
        tool="calculator",
        arguments={
            "a": 2,
            "b": 3,
            "operation": "add",
        },
        result="5",
        success=True,
    )

    assert (
        execution.tool
        == "calculator"
    )

    assert execution.success


def test_agent_trace():

    execution = ToolExecution(
        tool="calculator",
        arguments={},
        result="5",
        success=True,
    )

    trace = AgentTrace(
        tools=[
            "calculator",
        ],
        tool_results=[
            execution,
        ],
        steps=2,
    )

    assert trace.steps == 2

    assert trace.tools == [
        "calculator"
    ]


def test_agent_result():

    trace = AgentTrace()

    result = AgentResult(
        answer="Test answer",
        trace=trace,
    )

    assert (
        result.answer
        == "Test answer"
    )