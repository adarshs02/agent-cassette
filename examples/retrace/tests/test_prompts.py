"""Tests for agent prompt policies and specifications."""


def test_system_prompt_contains_repair_and_escalate_policies():
    """Verify the repair-vs-escalate policy is present in the system prompt."""
    from retrace.agent.prompts import SYSTEM_PROMPT

    assert "recoverable exactly" in SYSTEM_PROMPT
    assert "lost, missing or stale" in SYSTEM_PROMPT


def test_system_prompt_and_tools_are_generic():
    """Verify system prompt and tool descriptions don't mention specific domains."""
    from retrace.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

    # Forbidden words (case-insensitive) that are specific to prior faults
    forbidden = {"cloudpay", "cents", "raw_orders", "fx", "currency_code", "customer_id"}

    text_to_check = SYSTEM_PROMPT.lower()
    for word in forbidden:
        assert word not in text_to_check, f"SYSTEM_PROMPT must not mention '{word}'"

    for tool in TOOL_SCHEMAS:
        description = tool["description"].lower()
        for word in forbidden:
            assert word not in description, f"Tool '{tool['name']}' must not mention '{word}'"


def test_system_prompt_lines_are_readable():
    """Verify all SYSTEM_PROMPT lines fit within 100 character limit."""
    from retrace.agent.prompts import SYSTEM_PROMPT

    lines = SYSTEM_PROMPT.split("\n")
    for i, line in enumerate(lines, 1):
        assert len(line) <= 100, (
            f"Line {i} of SYSTEM_PROMPT is {len(line)} chars, exceeds 100: {line}"
        )
