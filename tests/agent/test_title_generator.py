"""Tests for agent.title_generator — auto-generated session titles."""

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from agent.title_generator import (
    generate_title,
    generate_title_from_history,
    auto_title_session,
    maybe_auto_title,
    _estimate_context_usage,
    _clean_title,
)


class TestCleanTitle:
    """Unit tests for _clean_title()."""

    def test_basic_title(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Debugging Python Import Errors"
        assert _clean_title(resp) == "Debugging Python Import Errors"

    def test_strips_quotes(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '"Setting Up Docker Environment"'
        assert _clean_title(resp) == "Setting Up Docker Environment"

    def test_strips_title_prefix(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Title: Kubernetes Pod Debugging"
        assert _clean_title(resp) == "Kubernetes Pod Debugging"

    def test_truncates_long_titles(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "A" * 100
        title = _clean_title(resp)
        assert len(title) == 80
        assert title.endswith("...")

    def test_returns_none_on_empty(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ""
        assert _clean_title(resp) is None


class TestEstimateContextUsage:
    """Tests for _estimate_context_usage()."""

    def test_empty_history(self):
        assert _estimate_context_usage([], 128000) == 0.0

    def test_zero_context_length(self):
        assert _estimate_context_usage([{"role": "user", "content": "hello"}], 0) == 0.0

    def test_returns_fraction(self):
        # 512000 chars / 4 = 128000 tokens, 128000/128000 = 1.0
        history = [{"role": "user", "content": "x" * 512000}]
        usage = _estimate_context_usage(history, 128000)
        assert usage == 1.0

    def test_capped_at_one(self):
        history = [{"role": "user", "content": "x" * 1000000}]
        usage = _estimate_context_usage(history, 128000)
        assert usage == 1.0

    def test_partial_usage(self):
        # ~64000 chars / 4 = 16000 tokens, 16000/128000 ≈ 0.125
        history = [{"role": "user", "content": "x" * 64000}]
        usage = _estimate_context_usage(history, 128000)
        assert 0.10 < usage < 0.15


class TestGenerateTitle:
    """Unit tests for generate_title() (first-exchange method)."""

    def test_returns_title_on_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Debugging Python Import Errors"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("help me fix this import", "Sure, let me check...")
            assert title == "Debugging Python Import Errors"

    def test_returns_none_on_exception(self):
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("no provider")):
            assert generate_title("question", "answer") is None

    def test_truncates_long_messages(self):
        """Long user/assistant messages should be truncated in the LLM request."""
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Short Title"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title("x" * 1000, "y" * 1000)

        user_content = captured_kwargs["messages"][1]["content"]
        assert len(user_content) < 1100  # 500 + 500 + formatting


class TestGenerateTitleFromHistory:
    """Tests for generate_title_from_history()."""

    def test_returns_title_from_conversation(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Docker Compose Setup"

        history = [
            {"role": "user", "content": "How do I set up docker compose?"},
            {"role": "assistant", "content": "Create a docker-compose.yml file..."},
            {"role": "user", "content": "What about networking?"},
            {"role": "assistant", "content": "Use the networks key..."},
        ]

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title_from_history(history)
            assert title == "Docker Compose Setup"

    def test_returns_none_on_empty_history(self):
        assert generate_title_from_history([]) is None

    def test_returns_none_on_exception(self):
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("fail")):
            history = [{"role": "user", "content": "hello"}]
            assert generate_title_from_history(history) is None


class TestAutoTitleSession:
    """Tests for auto_title_session() — the sync worker function."""

    def test_skips_if_no_session_db(self):
        auto_title_session(None, "sess-1", "hi", "hello")  # should not crash

    def test_skips_if_title_exists(self):
        db = MagicMock()
        db.get_session_title.return_value = "Existing Title"

        with patch("agent.title_generator.generate_title") as gen:
            auto_title_session(db, "sess-1", "hi", "hello")
            gen.assert_not_called()

    def test_generates_and_sets_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch("agent.title_generator.generate_title", return_value="New Title"):
            auto_title_session(db, "sess-1", "hi", "hello")
            db.set_session_title.assert_called_once_with("sess-1", "New Title")

    def test_uses_history_when_flag_set(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
        ]

        with patch("agent.title_generator.generate_title_from_history", return_value="History Title") as mock_hist:
            with patch("agent.title_generator.generate_title") as mock_exchange:
                auto_title_session(db, "sess-1", "hi", "hello",
                                   conversation_history=history, use_history=True)
                mock_hist.assert_called_once_with(history)
                mock_exchange.assert_not_called()
                db.set_session_title.assert_called_once_with("sess-1", "History Title")

    def test_skips_if_generation_fails(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch("agent.title_generator.generate_title", return_value=None):
            auto_title_session(db, "sess-1", "hi", "hello")
            db.set_session_title.assert_not_called()


class TestMaybeAutoTitle:
    """Tests for maybe_auto_title() — the fire-and-forget entry point."""

    def _default_config(self):
        """Return default auto_title config for patching."""
        return {"enabled": True, "context_threshold": 0.15, "max_turns": 4}

    def test_fires_on_first_exchange(self):
        """Should fire for conversations with <= 2 user messages (legacy fallback)."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        with patch("agent.title_generator._load_auto_title_config", return_value=self._default_config()):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title(db, "sess-1", "hello", "hi there", history)
                time.sleep(0.3)
                mock_auto.assert_called_once()
                # Should use first-exchange method (no history)
                args, kwargs = mock_auto.call_args
                assert kwargs.get("use_history") is False

    def test_fires_on_context_threshold(self):
        """Should fire when context usage exceeds the threshold."""
        db = MagicMock()
        db.get_session_title.return_value = None

        # Create history with enough content to exceed 15% of a 128K context
        history = [
            {"role": "user", "content": "x" * 25000},   # ~6250 tokens
            {"role": "assistant", "content": "y" * 25000},
            {"role": "user", "content": "z" * 25000},
        ]

        with patch("agent.title_generator._load_auto_title_config", return_value=self._default_config()):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title(
                    db, "sess-1", "message", "response", history,
                    context_length=128000,
                    last_prompt_tokens=0,
                )
                time.sleep(0.3)
                mock_auto.assert_called_once()
                args, kwargs = mock_auto.call_args
                assert kwargs.get("use_history") is True

    def test_fires_on_prompt_tokens_exceeding_threshold(self):
        """Should fire when last_prompt_tokens indicate threshold exceeded."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
        ]

        with patch("agent.title_generator._load_auto_title_config", return_value=self._default_config()):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                # 20000/128000 = ~15.6% > 15% threshold
                maybe_auto_title(
                    db, "sess-1", "msg2", "response", history,
                    context_length=128000,
                    last_prompt_tokens=20000,
                )
                time.sleep(0.3)
                mock_auto.assert_called_once()

    def test_fires_on_max_turns(self):
        """Should fire when user message count >= max_turns."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "resp3"},
            {"role": "user", "content": "msg4"},
        ]

        with patch("agent.title_generator._load_auto_title_config", return_value=self._default_config()):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title(
                    db, "sess-1", "msg4", "resp4", history,
                    context_length=128000,
                    last_prompt_tokens=5000,  # well under 15%
                )
                time.sleep(0.3)
                mock_auto.assert_called_once()

    def test_skips_if_not_yet_triggered(self):
        """Should NOT fire when: > 2 user messages, < max_turns, < threshold."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
        ]
        # user_msg_count = 2, but this is NOT <= 2 because the current message
        # is already in the history. Actually let's test with 3 user messages
        # but well under threshold and under max_turns.
        history = [
            {"role": "user", "content": "short1"},
            {"role": "assistant", "content": "short resp"},
            {"role": "user", "content": "short2"},
        ]

        config = {"enabled": True, "context_threshold": 0.50, "max_turns": 10}

        with patch("agent.title_generator._load_auto_title_config", return_value=config):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title(
                    db, "sess-1", "short2", "response", history,
                    context_length=128000,
                    last_prompt_tokens=100,  # very low
                )
                time.sleep(0.3)
                mock_auto.assert_not_called()

    def test_skips_if_disabled_in_config(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [{"role": "user", "content": "hello"}]

        config = {"enabled": False, "context_threshold": 0.15, "max_turns": 4}

        with patch("agent.title_generator._load_auto_title_config", return_value=config):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title(db, "sess-1", "hello", "response", history)
                time.sleep(0.3)
                mock_auto.assert_not_called()

    def test_skips_if_no_response(self):
        db = MagicMock()
        maybe_auto_title(db, "sess-1", "hello", "", [])  # empty response

    def test_skips_if_no_session_db(self):
        maybe_auto_title(None, "sess-1", "hello", "response", [])  # no db

    def test_skips_if_title_already_set(self):
        db = MagicMock()
        db.get_session_title.return_value = "Existing Title"
        history = [{"role": "user", "content": "hello"}]

        with patch("agent.title_generator._load_auto_title_config", return_value=self._default_config()):
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                maybe_auto_title(db, "sess-1", "hello", "response", history)
                time.sleep(0.3)
                mock_auto.assert_not_called()
