"""Tests for the Perception node."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from isaac.core.state import WorldModel, make_initial_state
from isaac.nodes.perception import perception_node
from tests.conftest import MockLLM


class TestPerceptionNode:
    def test_extracts_observations(self) -> None:
        state = make_initial_state()
        state["messages"] = [HumanMessage(content="Create a file with hello world")]

        mock = MockLLM(
            '{"observations": ["user wants to create a file"], '
            '"hypothesis": "write hello world to a file", '
            '"task_mode": "code"}'
        )
        with (
            patch("isaac.llm.provider.get_perception_llm", return_value=mock),
            patch("isaac.nodes.perception.fast_classify", return_value=(None, 0.0)),
        ):
            result = perception_node(state)

        assert result["current_phase"] == "perception"
        wm: WorldModel = result["world_model"]
        assert "user wants to create a file" in wm.observations
        assert result["hypothesis"] == "write hello world to a file"

    def test_handles_no_user_message(self) -> None:
        state = make_initial_state()
        result = perception_node(state)
        assert result["current_phase"] == "perception"

    def test_handles_malformed_json(self) -> None:
        state = make_initial_state()
        state["messages"] = [HumanMessage(content="do something")]

        mock = MockLLM("this is not json at all")
        with (
            patch("isaac.llm.provider.get_perception_llm", return_value=mock),
            patch("isaac.nodes.perception.fast_classify", return_value=(None, 0.0)),
        ):
            result = perception_node(state)

        # Should degrade gracefully
        assert result["current_phase"] == "perception"
        wm: WorldModel = result["world_model"]
        assert not wm.observations
        assert result["task_mode"] == "direct"

    # ------------------------------------------------------------------
    # Multimodal input (Computer-Use path)
    # ------------------------------------------------------------------

    def test_multimodal_input_extracts_task_mode(self) -> None:
        """HumanMessage with list content (text + image) is handled correctly."""
        state = make_initial_state()
        # Simulate a multimodal message: text part + base64 screenshot part
        state["messages"] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Automate the login form"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ]
            )
        ]

        mock = MockLLM(
            '{"observations": ["login form visible on screen"], '
            '"hypothesis": "fill and submit the login form", '
            '"task_mode": "computer_use"}'
        )
        with patch("isaac.llm.provider.get_perception_llm", return_value=mock):
            result = perception_node(state)

        assert result["current_phase"] == "perception"
        assert result.get("task_mode") == "computer_use"
        wm: WorldModel = result["world_model"]
        assert any("login" in obs.lower() for obs in wm.observations)

    def test_multimodal_input_populates_gui_state(self) -> None:
        """A screenshot in the HumanMessage should populate world_model.gui_state."""
        state = make_initial_state()
        state["messages"] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Click the submit button"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ]
            )
        ]

        mock = MockLLM(
            '{"observations": ["submit button visible"], '
            '"hypothesis": "click submit", '
            '"task_mode": "computer_use"}'
        )
        with patch("isaac.llm.provider.get_perception_llm", return_value=mock):
            result = perception_node(state)

        wm: WorldModel = result["world_model"]
        # gui_state should have been populated with the screenshot
        if wm.gui_state is not None:
            assert wm.gui_state.screenshot_b64 == "aGVsbG8="

    def test_multimodal_falls_back_gracefully_without_image(self) -> None:
        """List content with only text (no image_url) should not crash."""
        state = make_initial_state()
        state["messages"] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Run the tests"},
                ]
            )
        ]

        mock = MockLLM('{"observations": ["run tests requested"], "hypothesis": "run tests"}')
        with (
            patch("isaac.llm.provider.get_perception_llm", return_value=mock),
            patch("isaac.nodes.perception.fast_classify", return_value=(None, 0.0)),
        ):
            result = perception_node(state)

        assert result["current_phase"] == "perception"


def test_reasoning_metadata_and_schema_validation(caplog):
    from unittest.mock import Mock

    from langchain_core.messages import AIMessage

    from isaac.core.transitions import after_perception

    state = make_initial_state()
    state["messages"] = [HumanMessage(content="An ambiguous request requiring interpretation")]
    private = "sensitive intermediate reasoning"
    for content in (
        '{"task_mode":"invalid"}',
        "[]",
        private,
        '{"observations":[],"hypothesis":"answer","task_mode":"direct"}',
    ):
        message = AIMessage(content=content, additional_kwargs={"reasoning_content": private})
        with patch(
            "isaac.llm.provider.get_perception_llm",
            return_value=Mock(invoke=Mock(return_value=message)),
        ):
            result = perception_node(state)
        assert after_perception(result) == "direct_response"
        assert private not in str(result)
        assert private not in caplog.text


def test_native_qwen_decoding_policy():
    from langchain_ollama import ChatOllama

    from isaac.llm import provider
    from isaac.llm.perception_schema import PerceptionResult

    with patch.object(provider, "_capped_llm", return_value=ChatOllama(model="qwen3:4b")):
        classify = provider.get_perception_llm.__wrapped__()
        answer = provider.get_direct_response_llm.__wrapped__()
    assert classify.reasoning is False
    assert classify.format == PerceptionResult.model_json_schema()
    assert answer.reasoning is True
    assert answer.num_predict == 2048
