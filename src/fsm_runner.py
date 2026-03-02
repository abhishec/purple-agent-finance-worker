"""
fsm_runner.py
Finance-domain dynamic FSM for structured task execution.

Supports all states from dynamic_fsm synthesis:
  RETRIEVE, PARSE, ANALYZE, VALIDATE (finance-domain)
  DECOMPOSE, ASSESS, COMPUTE, POLICY_CHECK, MUTATE, SCHEDULE_NOTIFY (general)
  COMPLETE, ESCALATE, FAILED (terminal)

Parallel state groups: states listed together execute simultaneously.
Branching hints: if/else conditions injected as guidance for Claude.

All process types are synthesized by Haiku at first encounter via dynamic_fsm.py.
Fallback: RETRIEVE → ANALYZE → COMPLETE (3-state finance path).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FSMState(str, Enum):
    # Finance-domain states
    RETRIEVE        = "RETRIEVE"        # fetch external data
    PARSE           = "PARSE"           # extract structured data
    ANALYZE         = "ANALYZE"         # multi-step reasoning
    VALIDATE        = "VALIDATE"        # cross-check results
    # General states (kept for dynamic synthesis generality)
    DECOMPOSE       = "DECOMPOSE"       # break task into sub-tasks
    ASSESS          = "ASSESS"          # gather data via read-only tools
    COMPUTE         = "COMPUTE"         # run calculations (pure math)
    POLICY_CHECK    = "POLICY_CHECK"
    MUTATE          = "MUTATE"          # write/change state
    SCHEDULE_NOTIFY = "SCHEDULE_NOTIFY"
    # Terminal
    COMPLETE        = "COMPLETE"
    ESCALATE        = "ESCALATE"
    FAILED          = "FAILED"


# Finance fallback — used when synthesis fails or returns empty
_FINANCE_GENERAL: list[FSMState] = [
    FSMState.RETRIEVE,
    FSMState.ANALYZE,
    FSMState.COMPLETE,
]

# Default per-state instructions (synthesized instructions take priority)
_DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "RETRIEVE": (
        "Fetch all data needed for this task: financial documents, prices, filings, "
        "trade records, API responses. Use all available tools. Do not take write actions yet."
    ),
    "PARSE": (
        "Extract structured data from fetched documents, tables, or raw text. "
        "Identify key metrics, numbers, and categorical fields. Build a structured data view."
    ),
    "ANALYZE": (
        "Analyze the gathered data. Apply relevant formulas and models. "
        "Identify patterns, compute metrics, draw evidence-based conclusions. "
        "Use any registered computation tools (Black-Scholes, amortization, etc.)."
    ),
    "VALIDATE": (
        "Cross-check computed results against source data. "
        "Verify numerical consistency. Re-derive key values from first principles "
        "if confidence is low. Flag any discrepancies."
    ),
    "DECOMPOSE": (
        "Break this task into sub-tasks. Identify all entities, data sources, "
        "and computation steps required before starting."
    ),
    "ASSESS": (
        "Gather all required data via read-only tools. "
        "Do not take any write actions — only collect data."
    ),
    "COMPUTE": (
        "Run all required calculations now. Show your work clearly. "
        "Use registered computation tools if available. "
        "Prepare exact results with the required precision."
    ),
    "POLICY_CHECK": (
        "Verify all policy rules and thresholds against computed values. "
        "Do not proceed if any rule is violated."
    ),
    "MUTATE": (
        "Execute all required state changes via write tools. "
        "Be systematic. Log each action as you take it."
    ),
    "SCHEDULE_NOTIFY": (
        "Handle all notifications, scheduling, and follow-up actions. "
        "Confirm recipients and timing."
    ),
    "COMPLETE": (
        "Provide a complete, structured answer. Include all computed values, "
        "data sources, and confidence indicators. Match the requested output format exactly."
    ),
    "ESCALATE": "Escalation required. Explain clearly why and who must act.",
    "FAILED":   "Execution failed. Explain what went wrong and suggest next steps.",
}


def _to_fsm_state(name: str) -> FSMState:
    """Convert state name to FSMState enum, defaulting to ANALYZE for unknowns."""
    try:
        return FSMState(name)
    except ValueError:
        return FSMState.ANALYZE


def _states_from_definition(definition: dict) -> list[Any]:
    """
    Convert synthesized definition state list to FSMState list.
    Parallel groups (nested lists) are preserved as list[FSMState].
    """
    result: list[Any] = []
    for item in definition.get("states", []):
        if isinstance(item, str):
            result.append(_to_fsm_state(item))
        elif isinstance(item, list):
            group = [_to_fsm_state(s) for s in item if isinstance(s, str)]
            if group:
                result.append(group)   # list = parallel group
    if not result:
        return _FINANCE_GENERAL[:]
    # Ensure COMPLETE at end
    last = result[-1]
    if isinstance(last, list) or last != FSMState.COMPLETE:
        result.append(FSMState.COMPLETE)
    return result


def _flatten_states(states: list[Any]) -> list[FSMState]:
    """Flatten parallel groups into sequential list for index tracking."""
    flat: list[FSMState] = []
    for item in states:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _format_states(states: list[Any]) -> str:
    """Human-readable state sequence, e.g. 'RETRIEVE → [ANALYZE || COMPUTE] → COMPLETE'."""
    parts = []
    for item in states:
        if isinstance(item, list):
            parts.append("[" + " || ".join(s.value for s in item) + "]")
        else:
            parts.append(item.value)
    return " → ".join(parts)


@dataclass
class FSMContext:
    task_text: str
    session_id: str
    process_type: str
    current_state: FSMState = FSMState.RETRIEVE
    state_history: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    requires_hitl: bool = False


class FSMRunner:
    """
    Finance-domain FSM for dynamic task execution.
    Driven by synthesized definitions — no hardcoded process templates.

    Key behaviors:
    - build_phase_prompt() generates comprehensive prompt from all synthesized instructions
    - Parallel groups: all parallel state instructions combined into one prompt section
    - Branching hints: if/else logic injected as guidance for Claude
    - Fallback: RETRIEVE → ANALYZE → COMPLETE for unrecognized types
    """

    def __init__(
        self,
        task_text: str,
        session_id: str,
        process_type: str | None = None,
        checkpoint=None,
        definition: dict | None = None,
    ):
        from src.smart_classifier import _keyword_fallback
        ptype = process_type or _keyword_fallback(task_text)
        self.ctx = FSMContext(task_text=task_text, session_id=session_id, process_type=ptype)
        self._definition: dict | None = definition
        self._idx = 0

        if checkpoint:
            ptype_c = checkpoint.process_type
            self.ctx.process_type = ptype_c
            # Restore synthesized definition from cache on checkpoint
            if definition is None:
                try:
                    from src.dynamic_fsm import get_synthesized
                    self._definition = get_synthesized(ptype_c)
                except Exception:
                    pass
            self.states = (
                _states_from_definition(self._definition)
                if self._definition else _FINANCE_GENERAL[:]
            )
            self._idx = checkpoint.state_idx
            self.ctx.state_history = list(checkpoint.state_history)
            flat = _flatten_states(self.states)
            self.ctx.current_state = (
                flat[self._idx] if self._idx < len(flat) else FSMState.COMPLETE
            )
        else:
            self.states = (
                _states_from_definition(self._definition)
                if self._definition else _FINANCE_GENERAL[:]
            )
            flat = _flatten_states(self.states)
            self.ctx.current_state = flat[0] if flat else FSMState.RETRIEVE

    @property
    def current_state(self) -> FSMState:
        return self.ctx.current_state

    @property
    def process_type(self) -> str:
        return self.ctx.process_type

    @property
    def is_terminal(self) -> bool:
        return self.ctx.current_state in (FSMState.COMPLETE, FSMState.FAILED, FSMState.ESCALATE)

    def advance(self, data: dict | None = None) -> FSMState:
        if data:
            self.ctx.data.update(data)
        self.ctx.state_history.append(self.ctx.current_state.value)
        self._idx += 1
        flat = _flatten_states(self.states)
        self.ctx.current_state = (
            flat[self._idx] if self._idx < len(flat) else FSMState.COMPLETE
        )
        return self.ctx.current_state

    def build_phase_prompt(self) -> str:
        """
        Build comprehensive phase prompt for Claude.
        Includes ALL state instructions from the synthesized definition —
        Claude has the full workflow picture, not just the current state.
        Parallel groups and branching hints are surfaced as explicit guidance.
        """
        process = self.ctx.process_type.replace("_", " ").title()
        state = self.ctx.current_state

        recent = (self.ctx.state_history + [state.value])[-4:]
        prefix = "...→ " if len(self.ctx.state_history) > 3 else ""
        history_str = prefix + " → ".join(recent)

        lines = [
            f"## Finance Process: {process}",
            f"## Current Phase: {state.value}",
            f"## Phase History: {history_str}",
            "",
        ]

        if self._definition:
            state_instructions = self._definition.get("state_instructions", {})
            connector_hints   = self._definition.get("connector_hints", [])
            branches          = self._definition.get("branches", {})
            risk_level        = self._definition.get("risk_level", "medium")
            workflow_str      = _format_states(self.states)

            lines.append(f"WORKFLOW: {workflow_str}")
            lines.append(f"Risk Level: {risk_level}")
            if connector_hints:
                lines.append(f"Relevant tool prefixes: {', '.join(connector_hints)}")
            lines.append("")
            lines.append("EXECUTION INSTRUCTIONS BY PHASE:")

            for item in self.states:
                if isinstance(item, list):
                    # Parallel group — run all simultaneously
                    parallel_names = " || ".join(s.value for s in item)
                    lines.append(f"\n[PARALLEL — execute these simultaneously: {parallel_names}]")
                    for s in item:
                        instr = (
                            state_instructions.get(s.value)
                            or _DEFAULT_INSTRUCTIONS.get(s.value, "")
                        )
                        if instr:
                            lines.append(f"  {s.value}: {instr}")
                else:
                    instr = (
                        state_instructions.get(item.value)
                        or _DEFAULT_INSTRUCTIONS.get(item.value, "")
                    )
                    if instr:
                        lines.append(f"\n{item.value}:\n{instr}")

            # Branching hints — guide Claude on conditional paths
            if branches:
                lines.append("\nBRANCHING LOGIC (conditional execution paths):")
                for state_name, branch_def in branches.items():
                    if isinstance(branch_def, dict):
                        for condition, next_states in branch_def.items():
                            if isinstance(next_states, list):
                                path = " → ".join(str(s) for s in next_states)
                                lines.append(f"  After {state_name} if {condition}: {path}")

            synthesized = self._definition.get("_synthesized", False)
            tag = "Dynamic FSM (synthesized)" if synthesized else "Dynamic FSM (cached)"
            lines.append(f"\n[{tag} for '{self.ctx.process_type}']")

        else:
            # Fallback — no synthesized definition
            lines.append("WORKFLOW: RETRIEVE → ANALYZE → COMPLETE")
            lines.append("")
            lines.append("Execute this finance task through the following phases:")
            lines.append("")
            for s in _FINANCE_GENERAL:
                instr = _DEFAULT_INSTRUCTIONS.get(s.value, "")
                if instr:
                    lines.append(f"{s.value}:\n{instr}\n")

        return "\n".join(lines)

    def get_summary(self) -> dict:
        return {
            "process_type": self.ctx.process_type,
            "final_state": self.ctx.current_state.value,
            "state_history": self.ctx.state_history,
            "requires_hitl": self.ctx.requires_hitl,
            "escalation_reason": "",
        }
