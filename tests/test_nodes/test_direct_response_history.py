"""Direct prompt regression tests, extended for the SELENE voice language."""

from langchain_core.messages import AIMessage, HumanMessage

from isaac.nodes.direct_response import _build_direct_prompt


def test_direct_prompt_includes_prior_exchange() -> None:
    history = [HumanMessage(content="My name is Ada"), AIMessage(content="Hello Ada")]
    prompt = _build_direct_prompt("What is my name?", "", history)
    assert [message.content for message in prompt[-3:]] == [
        "My name is Ada",
        "Hello Ada",
        "What is my name?",
    ]


def test_direct_prompt_uses_the_voice_sessions_language() -> None:
    prompt = _build_direct_prompt(
        "What is the capital of Brazil?", "", response_language="English only"
    )
    assert "Reply in English only." in prompt[0].content
