"""Baseline unit tests for the intake agent.

Principle: test OUR code (tool wiring, dispatch routing, state shape), never the
model's judgement. The LLM is an external dependency, so it is faked here with
canned responses — these tests run in milliseconds and never touch Ollama.

Whether qwen3:8b *classifies correctly* is an eval concern, not a unit test, and
lives outside this file.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from denoiser.intake import main
from denoiser.intake.main import (
    State,
    call_model,
    cosntruct_graph,
    log_reason,
    preserve_signal,
)


class FakeChatModel:
    """Deterministic stand-in for the bound ChatOllama client.

    Returns queued responses in order, one per ``invoke`` call, and records the
    message lists it was called with so tests can assert on what the agent sent.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        return self._responses.pop(0)


def _ai_with_tool_call(name: str, args: dict, tool_id: str = "call_1") -> AIMessage:
    """Build an AIMessage that requests a single tool call, as Ollama would."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tool_id, "type": "tool_call"}],
    )


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """Patch the module-global ``ai_client`` with a queue-backed fake.

    Returns a factory so each test seeds its own canned response sequence.
    """

    def _install(responses: list[AIMessage]) -> FakeChatModel:
        client = FakeChatModel(responses)
        monkeypatch.setattr(main, "ai_client", client)
        return client

    return _install


# --- Tools: trivial, but they are the contract the dispatch relies on ---------


@pytest.mark.parametrize(
    "tool,args,expected",
    [
        (preserve_signal, {"summary": "rent due Friday"}, "Signal preserved: rent due Friday"),
        (log_reason, {"summary": "promo newsletter"}, "Signal skipped: promo newsletter"),
    ],
)
def test_tool_returns_marked_content(tool, args: dict, expected: str) -> None:
    assert tool.invoke(args) == expected


# --- call_model dispatch: this is where real agent bugs live ------------------


def test_call_model_no_tool_calls_returns_only_model_message(fake_client) -> None:
    fake_client([AIMessage(content="just a plain answer")])

    out = call_model({"email": {}, "messages": []})

    messages = out["messages"]
    assert len(messages) == 1
    assert messages[0].content == "just a plain answer"


def test_call_model_routes_preserve_signal(fake_client) -> None:
    client = fake_client([_ai_with_tool_call("preserve_signal", {"summary": "rent due"})])

    out = call_model({"email": {}, "messages": []})
    messages = out["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "Signal preserved: rent due"
    assert tool_messages[0].tool_call_id == "call_1"

    # generic dispatch runs the tool once, no extra model pass
    assert len(client.calls) == 1


def test_call_model_routes_log_reason(fake_client) -> None:
    client = fake_client([_ai_with_tool_call("log_reason", {"summary": "newsletter"})])

    out = call_model({"email": {}, "messages": []})
    messages = out["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "Signal skipped: newsletter"
    assert len(client.calls) == 1


def test_call_model_embeds_email_in_system_prompt(fake_client) -> None:
    client = fake_client([AIMessage(content="ok")])

    call_model({"email": {"email_message": "UNIQUE-MARKER-123"}, "messages": []})

    system_prompt = client.calls[0][0].content
    assert "UNIQUE-MARKER-123" in system_prompt


# --- Graph wiring -------------------------------------------------------------


def test_graph_compiles_with_assistant_entry_node() -> None:
    graph = cosntruct_graph()

    assert "assistant" in graph.get_graph().nodes


def test_graph_invoke_appends_agent_output_to_state(fake_client) -> None:
    fake_client([_ai_with_tool_call("log_reason", {"summary": "promo"})])
    graph = cosntruct_graph()

    result = graph.invoke(
        State(
            email={"email_message": "buy now!"},
            messages=[HumanMessage(content="Please categorise the email content")],
        )
    )

    # add_messages reducer keeps the input HumanMessage and appends agent output
    assert any(isinstance(m, HumanMessage) for m in result["messages"])
    assert any(
        isinstance(m, ToolMessage) and m.content == "Signal skipped: promo"
        for m in result["messages"]
    )


def test_call_model_skips_unknown_tool(fake_client) -> None:
    fake_client([_ai_with_tool_call("nonexistent_tool", {"summary": "x"})])

    out = call_model({"email": {}, "messages": []})

    # unknown tool name is skipped — only the model message remains, no ToolMessage
    assert not [m for m in out["messages"] if isinstance(m, ToolMessage)]
