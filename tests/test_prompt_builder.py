"""Tests for PromptBuilder and PromptTemplate."""

import json
from unittest.mock import MagicMock, patch

import pytest

from qpo.models import PromptTemplate
from qpo.pipeline.prompt_builder import PromptBuilder


AXES = ["formal_tone", "include_examples", "step_by_step", "explicit_schema", "role_framing", "error_handling"]
GOAL = "Extract structured appointment data from patient messages and return JSON."


class TestPromptTemplate:
    def test_render_base_only(self):
        t = PromptTemplate(base="Do the task.", modifiers={"formal_tone": "Be formal."})
        result = t.render({"formal_tone": 0})
        assert result == "Do the task."

    def test_render_with_active_modifier(self):
        t = PromptTemplate(base="Do the task.", modifiers={"formal_tone": "Be formal."})
        result = t.render({"formal_tone": 1})
        assert "Do the task." in result
        assert "Be formal." in result

    def test_render_multiple_active_modifiers(self):
        t = PromptTemplate(
            base="Base.",
            modifiers={"a": "Modifier A.", "b": "Modifier B.", "c": "Modifier C."},
        )
        result = t.render({"a": 1, "b": 0, "c": 1})
        assert "Modifier A." in result
        assert "Modifier B." not in result
        assert "Modifier C." in result

    def test_render_unknown_axis_ignored(self):
        t = PromptTemplate(base="Base.", modifiers={"known": "Known modifier."})
        result = t.render({"known": 1, "unknown_axis": 1})
        assert result.count("modifier") == 1

    def test_all_zero_returns_base(self):
        t = PromptTemplate(
            base="Just the base.",
            modifiers={ax: f"Modifier for {ax}." for ax in AXES},
        )
        result = t.render({ax: 0 for ax in AXES})
        assert result == "Just the base."

    def test_all_one_returns_base_plus_all_modifiers(self):
        t = PromptTemplate(
            base="Base.",
            modifiers={ax: f"Mod_{ax}." for ax in AXES},
        )
        result = t.render({ax: 1 for ax in AXES})
        for ax in AXES:
            assert f"Mod_{ax}." in result


class TestPromptBuilderPatternFallback:
    def test_pattern_fallback_produces_valid_template(self):
        builder = PromptBuilder(ollama_endpoint="http://localhost:19999")
        t = builder._pattern_template(GOAL, AXES)
        assert isinstance(t, PromptTemplate)
        assert len(t.base) > 0
        for ax in AXES:
            assert ax in t.modifiers
            assert len(t.modifiers[ax]) > 0

    def test_known_patterns_produce_meaningful_modifiers(self):
        builder = PromptBuilder()
        assert "professional" in builder._pattern_modifier("formal_tone").lower() or \
               "precise" in builder._pattern_modifier("formal_tone").lower()
        assert "example" in builder._pattern_modifier("include_examples").lower()
        assert "step" in builder._pattern_modifier("step_by_step").lower()

    def test_unknown_axis_gets_generic_modifier(self):
        builder = PromptBuilder()
        mod = builder._pattern_modifier("totally_novel_axis_xyz")
        assert len(mod) > 0
        assert "Totally Novel Axis Xyz" in mod or "totally novel axis xyz" in mod.lower()

    def test_fallback_used_when_ollama_unreachable(self):
        builder = PromptBuilder(ollama_endpoint="http://localhost:19999")
        t = builder.build_template(GOAL, AXES)
        assert isinstance(t, PromptTemplate)
        assert len(t.modifiers) == len(AXES)

    def test_render_produces_different_text_per_combination(self):
        builder = PromptBuilder(ollama_endpoint="http://localhost:19999")
        t = builder.build_template(GOAL, AXES)
        prompts = {
            t.render({ax: (i >> j) & 1 for j, ax in enumerate(AXES)})
            for i in range(min(16, 2 ** len(AXES)))
        }
        # At least half of 16 combinations should produce distinct prompts
        assert len(prompts) >= 8


class TestPromptBuilderLLMPath:
    def test_llm_response_parsed_correctly(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "response": json.dumps({
                "base": "Extract appointment data and return JSON.",
                "modifiers": {ax: f"Modifier for {ax}." for ax in AXES},
            }),
            "done": True,
        }
        with patch("requests.post", return_value=mock_resp):
            builder = PromptBuilder()
            t = builder.build_template(GOAL, AXES)

        assert t.base == "Extract appointment data and return JSON."
        for ax in AXES:
            assert ax in t.modifiers

    def test_partial_llm_response_filled_by_pattern(self):
        """LLM that misses some axes — missing ones get pattern fallback."""
        partial_axes = AXES[:3]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "response": json.dumps({
                "base": "Do the task.",
                "modifiers": {ax: f"Mod {ax}." for ax in partial_axes},
            }),
            "done": True,
        }
        with patch("requests.post", return_value=mock_resp):
            builder = PromptBuilder()
            t = builder.build_template(GOAL, AXES)

        for ax in AXES:
            assert ax in t.modifiers

    def test_llm_no_json_falls_back_to_pattern(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "Sorry I cannot help with that.", "done": True}
        with patch("requests.post", return_value=mock_resp):
            builder = PromptBuilder()
            t = builder.build_template(GOAL, AXES)

        # Should fall back without raising
        assert isinstance(t, PromptTemplate)
        assert len(t.modifiers) == len(AXES)
