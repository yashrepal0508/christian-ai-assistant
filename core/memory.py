MAX_HISTORY = 10  # keep last 10 turns (5 user + 5 assistant)


def trim_history(messages: list[dict]) -> list[dict]:
    """Keep only the most recent turns to stay within token limits."""
    return messages[-MAX_HISTORY:] if len(messages) > MAX_HISTORY else messages


def format_history_for_display(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["role"] in ("user", "assistant")]
