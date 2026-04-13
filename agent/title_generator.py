"""Auto-generate short session titles from conversation context.

Titles are generated once a configurable percentage of the context window has been
consumed, giving the LLM enough conversation history to produce a meaningful title.
The generation runs in a background thread so it never adds latency to the
user-facing reply.

Configuration (in ~/.hermes/config.yaml):

    auto_title:
        enabled: true            # set false to disable
        context_threshold: 0.15  # fraction of context window (default 15%)
        max_turns: 4             # also generate if this many user turns reached

If neither threshold is met, falls back to the legacy first-exchange heuristic
at 2 user messages.
"""

import logging
import threading
from typing import List, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Default: generate a title once 15% of context is used, or after 4 user turns,
# whichever comes first.  15% is enough to have real conversational context
# while still being early enough to be useful.
_DEFAULT_CONTEXT_THRESHOLD = 0.15
_DEFAULT_MAX_TURNS = 4

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for the following conversation. "
    "The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, "
    "no prefixes like 'Title:'."
)

_TITLE_PROMPT_HISTORY = (
    "Below is a conversation between a user and an AI assistant. "
    "Generate a short, descriptive title (3-7 words) that captures the main topic "
    "or intent of this conversation. Return ONLY the title text — no quotes, "
    "no punctuation at the end, no prefixes."
)


def _load_auto_title_config() -> dict:
    """Load auto-title config from hermes config.yaml.

    Returns a dict with keys: enabled, context_threshold, max_turns.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        at = cfg.get("auto_title", {})
        if isinstance(at, dict):
            return {
                "enabled": at.get("enabled", True),
                "context_threshold": float(at.get("context_threshold", _DEFAULT_CONTEXT_THRESHOLD)),
                "max_turns": int(at.get("max_turns", _DEFAULT_MAX_TURNS)),
            }
    except Exception:
        pass
    return {
        "enabled": True,
        "context_threshold": _DEFAULT_CONTEXT_THRESHOLD,
        "max_turns": _DEFAULT_MAX_TURNS,
    }


def _estimate_context_usage(conversation_history: list, context_length: int) -> float:
    """Rough estimate of context usage as a fraction 0.0-1.0.

    Uses the same fast estimation used elsewhere in the agent.
    """
    if not conversation_history or context_length <= 0:
        return 0.0
    # ~4 chars per token is a decent rough estimate
    total_chars = sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
        for m in conversation_history
    )
    estimated_tokens = total_chars // 4
    return min(estimated_tokens / context_length, 1.0)


def _clean_title(response) -> Optional[str]:
    """Extract and clean a title from an LLM response."""
    title = (response.choices[0].message.content or "").strip()
    title = title.strip('"\'')
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title if title else None


def generate_title(user_message: str, assistant_response: str, timeout: float = 30.0) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.
    """
    user_snippet = user_message[:500] if user_message else ""
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    messages = [
        {"role": "system", "content": _TITLE_PROMPT},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    try:
        response = call_llm(
            task="compression",  # reuse compression task config (cheap/fast model)
            messages=messages,
            max_tokens=30,
            temperature=0.3,
            timeout=timeout,
        )
        return _clean_title(response)
    except Exception as e:
        logger.debug("Title generation failed: %s", e)
        return None


def generate_title_from_history(
    conversation_history: list,
    timeout: float = 30.0,
) -> Optional[str]:
    """Generate a session title from the full conversation history.

    Produces a richer title because it has more context to work with.
    Truncates the history sent to the LLM to keep the request cheap.
    """
    if not conversation_history:
        return None

    # Build a compact representation of the conversation.
    # Keep it under ~2000 chars for the LLM (cheap model, fast response).
    parts = []
    budget = 2000
    for m in conversation_history:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        # Truncate individual messages
        snippet = content[:300] if len(content) > 300 else content
        parts.append(f"{role.title()}: {snippet}")
        budget -= len(snippet) + len(role) + 4
        if budget <= 0:
            break

    if not parts:
        return None

    conversation_text = "\n".join(parts)

    messages = [
        {"role": "system", "content": _TITLE_PROMPT_HISTORY},
        {"role": "user", "content": conversation_text},
    ]

    try:
        response = call_llm(
            task="compression",
            messages=messages,
            max_tokens=30,
            temperature=0.3,
            timeout=timeout,
        )
        return _clean_title(response)
    except Exception as e:
        logger.debug("Title generation from history failed: %s", e)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list = None,
    use_history: bool = False,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called in a background thread after a turn completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    """
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    # Use the richer history-based generation when we have enough context,
    # otherwise fall back to the first-exchange method.
    if use_history and conversation_history:
        title = generate_title_from_history(conversation_history)
    else:
        title = generate_title(user_message, assistant_response)

    if not title:
        return

    try:
        session_db.set_session_title(session_id, title)
        logger.debug("Auto-generated session title: %s", title)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    context_length: int = 0,
    last_prompt_tokens: int = 0,
) -> None:
    """Fire-and-forget title generation based on context usage.

    Generates a title when one of these conditions is met:
    - Context usage >= auto_title.context_threshold (default 15%)
    - User turn count >= auto_title.max_turns (default 4)
    - Legacy: first 2 exchanges (fallback, always works)

    Skips if the session already has a title or if generation fails.
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    cfg = _load_auto_title_config()
    if not cfg.get("enabled", True):
        return

    # Count user messages in history
    user_msg_count = sum(
        1 for m in (conversation_history or [])
        if m.get("role") == "user"
    )

    # Check if we already have a title — quick check before any expensive work
    try:
        if session_db.get_session_title(session_id):
            return
    except Exception:
        return

    # Determine if we should generate a title
    should_generate = False
    use_history = False
    reason = ""

    # Condition 1: context usage threshold exceeded
    if context_length > 0:
        # Use the accurate prompt tokens if available, otherwise estimate
        if last_prompt_tokens > 0:
            usage = last_prompt_tokens / context_length
        else:
            usage = _estimate_context_usage(conversation_history, context_length)

        if usage >= cfg["context_threshold"]:
            should_generate = True
            use_history = True
            reason = f"context usage {usage:.0%} >= {cfg['context_threshold']:.0%}"
    else:
        # No context_length info — estimate from history size
        usage = _estimate_context_usage(conversation_history, 128000)
        if usage >= cfg["context_threshold"]:
            should_generate = True
            use_history = True
            reason = f"estimated context usage {usage:.0%} >= {cfg['context_threshold']:.0%}"

    # Condition 2: max turns reached
    if not should_generate and user_msg_count >= cfg["max_turns"]:
        should_generate = True
        use_history = True
        reason = f"user turns ({user_msg_count}) >= max ({cfg['max_turns']})"

    # Condition 3 (legacy): first 2 exchanges
    if not should_generate and user_msg_count <= 2:
        should_generate = True
        use_history = False  # Not enough history for rich title
        reason = f"first exchange (turns={user_msg_count})"

    if not should_generate:
        return

    logger.debug(
        "Triggering auto-title for session %s: %s",
        session_id[:16], reason,
    )

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        kwargs={
            "conversation_history": conversation_history,
            "use_history": use_history,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
