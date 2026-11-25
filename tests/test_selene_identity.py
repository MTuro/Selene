"""Regression coverage for the SELENE identity through the actual graph."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from isaac.config.settings import get_settings
from isaac.core.graph import build_graph
from isaac.identity.address import normalize_selene_address
from isaac.llm.prompts import perception_prompt
from isaac.memory.world_model import WorldModel
from isaac.nodes.guard import GuardResult


@pytest.fixture(autouse=True)
def stub_guard_model(monkeypatch):
    from isaac.nodes import guard

    monkeypatch.setattr(guard.PromptInjectionGuard, "_llm_analysis", lambda *_: GuardResult())


@pytest.mark.parametrize(
    "question",
    [
        "What's your name?",
        "Selene, what's your name?",
        "Selene, como você se chama?",
        "Celene, como voce se chama?",
    ],
)
def test_selene_answers_name_questions_in_english(monkeypatch, question):
    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    response_model = Mock(
        side_effect=AssertionError("A name question should use the active identity")
    )
    monkeypatch.setattr("isaac.llm.provider.get_direct_response_llm", response_model)

    state = build_graph().invoke({"messages": [HumanMessage(content=question)]})

    assert state["task_mode"] == "direct"
    assert state["messages"][-1].content == "My name is SELENE."
    response_model.assert_not_called()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Why are you called Selene?",
            "My name comes from the Greek word for Moon. "
            "In Greek mythology, Selene was the goddess and personification of the Moon. "
            "Marcos chose the name because of that connection.",
        ),
        ("Who named you?", "Marcos named me Selene."),
    ],
)
def test_selene_answers_name_origin_questions_without_history(monkeypatch, question, expected):
    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    model = Mock(
        side_effect=AssertionError("An identity question should use the permanent profile")
    )
    monkeypatch.setattr("isaac.llm.provider.get_direct_response_llm", model)
    monkeypatch.setattr("isaac.llm.provider.get_perception_llm", model)

    result = build_graph().invoke({"messages": [HumanMessage(content=question)]})

    assert result["task_mode"] == "direct"
    assert result["messages"][-1].content == expected
    model.assert_not_called()


def test_guard_handles_exact_name_questions_without_semantic_false_positives(monkeypatch):
    from isaac.nodes.guard import PromptInjectionGuard, guard_node

    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    deep_check = Mock(return_value=GuardResult(suspicion_score=0.9, blocked=True))
    monkeypatch.setattr(PromptInjectionGuard, "_llm_analysis", deep_check)
    question = "Selene, como você se chama?"
    assert not guard_node({"messages": [HumanMessage(content=question)]})["guard_blocked"]
    deep_check.assert_not_called()

    extended = HumanMessage(content=f"{question} Extra text.")
    assert guard_node({"messages": [extended]})["guard_blocked"]
    multimodal = HumanMessage(content=[{"type": "text", "text": question}])
    assert guard_node({"messages": [multimodal]})["guard_blocked"]
    assert deep_check.call_count == 2


@pytest.mark.parametrize("profile", ["selene", "isaac"])
def test_conversation_uses_the_active_identity(monkeypatch, capsys, profile):
    monkeypatch.setattr(get_settings(), "identity_profile", profile)
    llm = Mock()
    llm.stream.return_value = iter(
        [
            AIMessage(content="I am I.S."),
            AIMessage(
                content="A.A.C., the Intelligent System for Autonomous Action and Cognition."
            ),
        ]
    )
    monkeypatch.setattr("isaac.llm.provider.get_direct_response_llm", lambda: llm)

    result = build_graph().invoke({"messages": [HumanMessage(content="Who are you?")]})
    system = llm.stream.call_args.args[0][0].content
    perception = perception_prompt("An ambiguous request", WorldModel())[0].content
    response = result["messages"][-1].content

    if profile == "selene":
        assert response == "I am SELENE."
        assert "I.S.A.A.C." not in system + perception + capsys.readouterr().out
        assert "You are SELENE." in system and "You are SELENE." in perception
        assert "Selene (Σελήνη), comes from Ancient Greek" in system
        assert "goddess and personification of the Moon" in perception
    else:
        assert response.startswith("I am I.S.A.A.C.")
        assert "You are I.S.A.A.C." in system and "You are I.S.A.A.C." in perception
        assert "Σελήνη" not in system + perception


def test_clarification_uses_the_selene_identity(monkeypatch):
    from isaac.nodes.clarification import _formulate_question

    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="I'm ISAAC. Could you clarify, Marcos?")
    monkeypatch.setattr("isaac.llm.provider.get_llm", lambda _tier: llm)

    question = _formulate_question({"messages": [HumanMessage(content="Faça isso.")]})

    assert question == "I'm SELENE. Could you clarify, Marcos?"
    system = llm.invoke.call_args.args[0][0].content
    assert "You are SELENE." in system and "English" in system
    assert "I.S.A.A.C." not in system
    assert "Marcos chose your name because of its connection to the Moon." in system


def test_agent_uses_the_selene_identity(monkeypatch):
    from isaac.agents.agent_loop import AgentLoop

    monkeypatch.setattr(get_settings(), "identity_profile", "selene")
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="I am I.S.A.A.C.")

    result = AgentLoop(tools=[], llm=llm).run("Introduce yourself.")

    assert result.output == "I am SELENE."
    system = llm.invoke.call_args.args[0][0].content
    assert "You are SELENE." in system and "English" in system
    assert "I.S.A.A.C." not in system


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("Selene, what's your name?", "SELENE, what's your name?"),
        ("Hey Celine, what's your name?", "Hey SELENE, what's your name?"),
        ("Celene como você se chama?", "SELENE como você se chama?"),
        ("Tell me about Celine Dion.", "Tell me about Celine Dion."),
        ("Selena Gomez is a singer.", "Selena Gomez is a singer."),
        ("Selena, a singer, won the award.", "Selena, a singer, won the award."),
        ("I am talking to Selena, not you.", "I am talking to Selena, not you."),
    ],
)
def test_normalization_only_changes_a_leading_address(original, expected):
    assert normalize_selene_address(original) == expected
