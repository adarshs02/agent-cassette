"""Tests for agent prompt policies and specifications."""


def test_system_prompt_contains_repair_and_escalate_policies():
    """Verify the repair-vs-escalate policy is present in the system prompt."""
    from retrace.agent.prompts import SYSTEM_PROMPT

    assert "recoverable exactly" in SYSTEM_PROMPT
    assert "lost, missing or stale" in SYSTEM_PROMPT


def test_system_prompt_contains_load_bearing_sentences():
    """Verify key policy sentences are in the system prompt."""
    from retrace.agent.prompts import SYSTEM_PROMPT

    assert "Never impute or forward-fill to hide it" in SYSTEM_PROMPT
    # Normalize whitespace to handle line breaks
    normalized = " ".join(SYSTEM_PROMPT.split())
    assert "Repair this even if the root cause is an upstream contract violation" in normalized


def test_system_prompt_and_tools_are_generic():
    """Verify system prompt and tool descriptions don't mention specific faults.

    Derives forbidden terms from the fault registry (fault names, field names,
    table names) and processor names. Handles collisions for generic English
    words (amount, currency) by checking whole-word boundaries.
    """
    import re

    from retrace.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS
    from retrace.faults import fault_names, get_fault

    # Collect all terms to forbid
    forbidden_substrings = {
        "cloudpay_v2",
        "legacy_pos",
        "cloudpay",
        "cents",
    }

    # Derive terms from fault registry
    for fault_name in fault_names():
        fault = get_fault(fault_name)
        # Add fault name
        forbidden_substrings.add(fault_name)

        # Add field name if not None
        if fault.ground_truth.field is not None:
            forbidden_substrings.add(fault.ground_truth.field)

        # Add alt_fields
        for alt_field in fault.ground_truth.alt_fields:
            if alt_field is not None:
                forbidden_substrings.add(alt_field)

        # Add table part from asset
        if fault.ground_truth.asset is not None:
            table_part = fault.ground_truth.asset.split(".")[-1]
            forbidden_substrings.add(table_part)

    # Terms that must match whole words (generic English words)
    whole_word_terms = {"amount", "currency"}

    # Check SYSTEM_PROMPT
    prompt_text_lower = SYSTEM_PROMPT.lower()
    for term in forbidden_substrings:
        term_lower = term.lower()
        if term in whole_word_terms:
            # Match as whole word only
            if re.search(rf"\b{re.escape(term_lower)}\b", prompt_text_lower):
                raise AssertionError(
                    f"SYSTEM_PROMPT contains whole-word generic term '{term}' (collision)"
                )
        else:
            # Match as substring
            assert term_lower not in prompt_text_lower, f"SYSTEM_PROMPT must not mention '{term}'"

    # Check all tool descriptions
    for tool in TOOL_SCHEMAS:
        description_lower = tool["description"].lower()
        for term in forbidden_substrings:
            term_lower = term.lower()
            if term in whole_word_terms:
                # Match as whole word only
                if re.search(rf"\b{re.escape(term_lower)}\b", description_lower):
                    raise AssertionError(
                        f"Tool '{tool['name']}' contains whole-word generic"
                        f" term '{term}' (collision)"
                    )
            else:
                # Match as substring
                assert term_lower not in description_lower, (
                    f"Tool '{tool['name']}' must not mention '{term}'"
                )


def test_confirm_root_cause_description_mentions_revision_and_keeps_asset_guidance():
    """The description must document revision support and keep the asset-format sentence."""
    from retrace.agent.prompts import TOOL_SCHEMAS

    tool = next(t for t in TOOL_SCHEMAS if t["name"] == "confirm_root_cause")
    assert (
        "Can be called again to revise the root cause before a repair passes."
        in tool["description"]
    )
    assert tool["description"].endswith(
        "asset: schema-qualified table name (schema.table) or its DataHub URN; field: column name."
    )


def test_system_prompt_lines_are_readable():
    """Verify all SYSTEM_PROMPT lines fit within 100 character limit."""
    from retrace.agent.prompts import SYSTEM_PROMPT

    lines = SYSTEM_PROMPT.split("\n")
    for i, line in enumerate(lines, 1):
        assert len(line) <= 100, (
            f"Line {i} of SYSTEM_PROMPT is {len(line)} chars, exceeds 100: {line}"
        )


def test_system_prompt_defines_root_cause_asset():
    """Verify the system prompt defines root-cause asset as the originating table."""
    from retrace.agent.prompts import SYSTEM_PROMPT

    normalized = " ".join(SYSTEM_PROMPT.split())
    assert "most upstream table where the bad data or defect originates" in normalized


def test_root_cause_and_escalate_tools_clarify_asset_origin():
    """Verify tools clarify the asset is the originating, upstream table."""
    from retrace.agent.prompts import TOOL_SCHEMAS

    confirm_tool = next(t for t in TOOL_SCHEMAS if t["name"] == "confirm_root_cause")
    escalate_tool = next(t for t in TOOL_SCHEMAS if t["name"] == "escalate_upstream")

    assert "not the file you patch" in confirm_tool["description"]
    assert "not the file you patch" in escalate_tool["description"]
