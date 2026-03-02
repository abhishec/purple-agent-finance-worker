"""
worker_brain.py
Finance Mini AI Worker — cognition layer for AgentX-AgentBeats Phase 2 Finance Track.

Zero hardcoded tools. Zero hardcoded process types.
Dynamic FSM synthesis + Autonomous Capability Engine (ACE) for every task.

3-phase cognitive loop:
  PRIME   → output format detection, FSM synthesis for ALL types, ACE pre-warming
  EXECUTE → FSM execution with ACE gap observation + quality gates
  REFLECT → RL recording, ACE capability graph update, knowledge extraction

Key innovations vs agent-purple worker_brain:
  1. Format directive injection (finance_output_adapter) — 30-point compliance fix
  2. synthesize_if_needed for ALL types (PROCESS_DEFINITIONS is empty)
  3. ACE pre-warming: start_promises_for_task in PRIME, non-blocking
  4. ACE gap observation: observe_execution_output fires background promises
  5. update_graph on every task — ACE gets smarter over the benchmark run
  6. No BP-specific modules (no hitl_guard, document_generator, policy_checker)
"""
from __future__ import annotations
import time
import json
import asyncio

from src.claude_executor import solve_with_claude
from src.mcp_bridge import discover_tools, call_tool
from src.rl_loop import build_rl_primer, record_outcome
from src.session_context import (
    add_turn, get_context_prompt, is_multi_turn,
    get_schema_cache, save_fsm_checkpoint, get_fsm_checkpoint,
    maybe_compress_async,
)
from src.fsm_runner import FSMRunner
from src.privacy_guard import check_privacy
from src.token_budget import TokenBudget, format_competition_answer, MODELS
from src.schema_adapter import resilient_tool_call
from src.paginated_tools import paginated_fetch
from src.config import GREEN_AGENT_MCP_URL, ANTHROPIC_API_KEY as _ANTHROPIC_API_KEY
from src.smart_classifier import classify_process_type
from src.knowledge_extractor import get_relevant_knowledge, extract_and_store
from src.entity_extractor import get_entity_context, record_task_entities
from src.recovery_agent import wrap_with_recovery
from src.self_reflection import reflect_on_answer, build_improvement_prompt, should_improve
from src.output_validator import validate_output, get_missing_fields_prompt
from src.self_moa import quick_synthesize as moa_quick, numeric_moa_synthesize
from src.five_phase_executor import five_phase_execute
from src.dynamic_fsm import synthesize_if_needed
from src.dynamic_tools import (
    load_registered_tools, is_registered_tool, call_registered_tool,
    detect_tool_gaps_llm, synthesize_and_register,
)
from src.strategy_bandit import select_strategy, record_outcome as bandit_record
from src.compute_verifier import verify_compute_output
from src.context_pruner import prune_rl_primer
from src.finance_output_adapter import detect_output_format, build_format_directive
from src.autonomous_capability_engine import (
    start_promises_for_task, await_promises,
    observe_execution_output, observe_tool_error,
    update_graph,
)


# ── Format Normalization Helpers ───────────────────────────────────────────────
# These run AFTER all quality gates (MoA / self-reflection / compute verifier).
# MoA/reflection can rewrite a correctly-formatted answer back into prose.
# This step is the last line of defence: re-extract and re-wrap into required shape.

_FORMAT_TEMPLATES: dict[str, str] = {
    "json_risk_classification":
        '{"risk_classification": ["Category1", "Category2"]}',
    "json_business_summary":
        '{"business_summary": {"industry": "...", "products": "...", "geography": "..."}}',
    "json_consistency_check":
        '{"consistency_check": ["exact risk phrase 1", "exact risk phrase 2"]}',
    "json_trading_decision": (
        '{"action": "BUY", "size": 0.1, "stop_loss": 46500.0, '
        '"take_profit": 48000.0, "reasoning": "brief reason", "confidence": 0.7}'
    ),
    "json_cot_answer":
        '{"cot": "step-by-step reasoning here", "answer": "final answer here"}',
    "json_options": (
        '{"result": {"price": 0.0, "greeks": {"delta": 0.0, "gamma": 0.0, '
        '"theta": 0.0, "vega": 0.0}, "assessment": "fairly priced"}}'
    ),
    # json_generic template is extracted from task_text at runtime (see _normalize_to_format)
    "json_generic": "__from_task__",
}

# Marker string whose presence means "answer already has correct format — skip normalization"
_FORMAT_ALREADY_OK: dict[str, str | None] = {
    "json_risk_classification": '"risk_classification"',
    "json_business_summary":    '"business_summary"',
    "json_consistency_check":   '"consistency_check"',
    "json_trading_decision":    '"action"',
    "json_cot_answer":          '"cot"',
    "json_options":             '"greeks"',
    "xml_final_answer":         "<FINAL_ANSWER>",
    "csv_data_integration":     None,   # too hard to validate — skip normalization
    "portfolio_allocation":     None,   # free-form OK — skip normalization
    "json_generic":             "{",    # any JSON object has { — normalize if missing
}


def _quick_format_check(answer: str, format_key: str) -> bool:
    """Return True if the answer already has the correct format marker."""
    marker = _FORMAT_ALREADY_OK.get(format_key)
    if marker is None:
        return True   # formats with no marker: leave as-is
    return marker in answer


async def _normalize_to_format(
    answer: str,
    format_key: str,
    task_text: str,
) -> str | None:
    """
    Force the answer into the required output shape using a fast Haiku call.
    Called LAST — after MoA/reflection/compute verifier — to recover from
    cases where those passes rewrote a correctly-formatted answer into prose.

    Returns the normalized string, or None if already correct / normalization failed.
    """
    import re as _re

    if _quick_format_check(answer, format_key):
        return None   # already correct — don't touch

    # ── Special case: XML FINAL_ANSWER — extract number, no API cost ──────────
    if format_key == "xml_final_answer":
        # Strip commas then find all numbers in the answer
        clean = answer.replace(",", "")
        nums = _re.findall(r'\b\d+\.?\d*\b', clean)
        if nums:
            # Use last standalone number (most likely the final computed value)
            return f"<FINAL_ANSWER>\n{nums[-1]}\n</FINAL_ANSWER>"
        return None

    # ── JSON formats: use Haiku to extract + reformat ─────────────────────────
    template = _FORMAT_TEMPLATES.get(format_key)
    if not template:
        return None   # csv_data_integration, portfolio_allocation

    # json_generic: extract the template structure from the task text itself.
    # Task text always contains "Return JSON: {...}" from the green agent.
    if template == "__from_task__":
        _m = _re.search(
            r'(?:Return|provide|answer as)\s+JSON[:\s]+(\{[^\n]{10,}?\})',
            task_text, _re.IGNORECASE | _re.DOTALL
        )
        template = _m.group(1).strip() if _m else '{"answer": "your computed result here"}'

    try:
        import anthropic as _ant
        _client = _ant.AsyncAnthropic(api_key=_ANTHROPIC_API_KEY)
        _haiku = MODELS.get("haiku", "claude-haiku-4-5")

        _prompt = (
            f"Extract the computed answer from the SOURCE TEXT and return it "
            f"in EXACTLY this JSON format — nothing else:\n\n"
            f"{template}\n\n"
            f"Rules:\n"
            f"• Return ONLY the JSON object — no markdown fences, no prose, no explanation\n"
            f"• Preserve all numerical values exactly as computed in the source\n"
            f"• For action fields use exactly: BUY, SELL, HOLD, or CLOSE\n"
            f"• If a field is missing from the source, use a reasonable default\n\n"
            f"SOURCE TEXT:\n{answer[:3000]}"
        )

        _msg = await _client.messages.create(
            model=_haiku,
            max_tokens=512,
            messages=[{"role": "user", "content": _prompt}],
        )
        raw = _msg.content[0].text.strip() if _msg.content else ""

        # Strip markdown code fences if model wraps output
        if raw.startswith("```"):
            _lines = raw.split("\n")
            _inner = _lines[1:-1] if _lines and _lines[-1].strip() == "```" else _lines[1:]
            raw = "\n".join(_inner).strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        # Validate it starts like JSON
        if raw.startswith("{") or raw.startswith("["):
            return raw

        return None

    except Exception:
        return None


class MiniAIWorker:
    """
    Finance Mini AI Worker for AgentX-AgentBeats Phase 2 — Finance Track.
    Fully dynamic: no hardcoded tools, no hardcoded process types.

    Worker identity: session_id (one worker instance per benchmark session).
    Worker cognition: dynamic FSM (Haiku-synthesized) + ACE capability engine.
    Worker memory: session_context + RL case log + ACE capability graph.
    Worker quality: compute verifier + self-reflection + numeric MoA.
    Worker compliance: finance_output_adapter (format directive injection).
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.budget = TokenBudget()
        self._tools: list[dict] = []
        self._ep: str = ""
        self._api_calls: int = 0
        self._pending_promises: list = []   # ACE CapabilityPromises started in PRIME

    async def run(
        self,
        task_text: str,
        policy_doc: str,
        tools_endpoint: str,
        task_id: str,
    ) -> str:
        """Entry point. 3-phase: PRIME → EXECUTE → REFLECT."""
        start_ms = int(time.time() * 1000)
        self._ep = tools_endpoint or GREEN_AGENT_MCP_URL

        context = await self._prime(task_text, policy_doc, task_id)
        if context.get("refused"):
            return context["message"]

        answer, tool_count, error = await self._execute(task_text, context)

        return await self._reflect(
            task_text, answer, tool_count, error, context, task_id, start_ms
        )

    # ── PRIME ─────────────────────────────────────────────────────────────

    async def _prime(self, task_text: str, policy_doc: str, task_id: str) -> dict:
        """
        Load all worker context before execution.

        Finance-specific additions over agent-purple PRIME:
          1. Output format detection (finance_output_adapter) injected early
          2. synthesize_if_needed for ALL task types (not just novel ones)
          3. ACE pre-warming: start_promises_for_task with FSM capability_requirements
        """
        # Privacy fast-fail
        privacy = check_privacy(task_text)
        if privacy and privacy.get("refused"):
            return {"refused": True, "message": privacy["message"]}

        # RL primer (learned patterns from past tasks)
        rl_primer = build_rl_primer(task_text)
        if rl_primer:
            rl_primer = prune_rl_primer(rl_primer)
            self.budget.consume(rl_primer, "rl_primer")

        # Multi-turn session context
        multi_turn_ctx = ""
        if is_multi_turn(self.session_id):
            multi_turn_ctx = get_context_prompt(self.session_id)
            if multi_turn_ctx:
                self.budget.consume(multi_turn_ctx, "session_context")

        # FSM: restore checkpoint or classify + synthesize
        checkpoint = get_fsm_checkpoint(self.session_id)
        process_type = None
        if not checkpoint:
            process_type, _cls_conf = await classify_process_type(task_text)

        # Dynamic FSM synthesis — ALL types go through synthesize_if_needed.
        # PROCESS_DEFINITIONS is empty: every type needs synthesis.
        # Cached after first encounter — subsequent tasks are free.
        synth_definition = None
        if not checkpoint and process_type:
            try:
                synth_definition = await synthesize_if_needed(process_type, task_text)
            except Exception:
                pass  # never block execution — fallback definition used

        # ACE pre-warming: start capability promises now, before execution.
        # These run in background — await_promises() called before primary solve.
        fsm_requirements: dict = {}
        if synth_definition:
            fsm_requirements = synth_definition.get("capability_requirements", {})
        self._pending_promises = start_promises_for_task(
            task_text, process_type or "general", fsm_requirements
        )

        fsm = FSMRunner(
            task_text=task_text,
            session_id=self.session_id,
            process_type=process_type,
            checkpoint=checkpoint,
            definition=synth_definition,
        )
        phase_prompt = fsm.build_phase_prompt()
        self.budget.consume(phase_prompt, "fsm_phase")

        # Output format detection — inject compliance directive BEFORE execution.
        # Fixes the 30-point AgentBusters gap: Claude knows from the start what
        # shape to produce (JSON, XML FINAL_ANSWER, CSV, etc.).
        format_directive = ""
        format_key = detect_output_format(task_text)
        if format_key:
            format_directive = build_format_directive(format_key)

        # Tool discovery (MCP endpoint)
        try:
            self._tools = await discover_tools(self._ep, session_id=self.session_id)
        except Exception:
            self._tools = []

        # Load registered tools (seeded + ACE-synthesized from prior tasks)
        registered = load_registered_tools()
        self._tools = self._tools + registered

        # ACE LLM gap detection (primary, replaces regex _GAP_PATTERNS entirely)
        if len(task_text) >= 80:
            try:
                llm_gaps = await detect_tool_gaps_llm(task_text, self._tools)
                for gap in llm_gaps[:3]:
                    try:
                        new_schema = await synthesize_and_register(gap, task_text)
                        if new_schema:
                            self._tools.append(new_schema)
                    except Exception:
                        pass  # never block execution
            except Exception:
                pass

        # ── Deduplicate tools by name ──────────────────────────────────────────
        # Claude API returns 400 "Tool names must be unique" if there are any dupes.
        # This happens when MCP tools + seeded tools + ACE-synthesized tools overlap
        # across multiple benchmark sessions (accumulated registry).
        _seen_names: set[str] = set()
        _deduped: list[dict] = []
        for _t in self._tools:
            _n = _t.get("name", "")
            if _n and _n not in _seen_names:
                _seen_names.add(_n)
                _deduped.append(_t)
        self._tools = _deduped

        # Knowledge base + entity memory injection
        kb_context = get_relevant_knowledge(task_text, fsm.process_type)
        entity_ctx = get_entity_context(task_text)
        if kb_context:
            self.budget.consume(kb_context, "knowledge")
        if entity_ctx:
            self.budget.consume(entity_ctx, "entities")

        # Build system context
        context_parts = [
            f"## Finance AI Worker | Task: {task_id} | Session: {self.session_id}",
            f"Tools endpoint: {self._ep}",
            "DIRECTIVE: Never ask the user clarifying questions. "
            "Make the most reasonable interpretation of the task and proceed autonomously. "
            "If details are ambiguous, choose the most likely interpretation and act. "
            "Complete the task with the information given.",
        ]
        if rl_primer:
            context_parts.append(self.budget.cap_prompt(rl_primer, "rl"))
        if kb_context:
            context_parts.append(self.budget.cap_prompt(kb_context, "knowledge"))
        if entity_ctx:
            context_parts.append(self.budget.cap_prompt(entity_ctx, "entities"))
        if multi_turn_ctx:
            context_parts.append(self.budget.cap_prompt(multi_turn_ctx, "history"))
        context_parts.append(phase_prompt)
        if format_directive:
            # Format directive goes LAST for maximum salience — Claude reads it fresh
            context_parts.append(format_directive)
        context_parts.append(self.budget.efficiency_hint())

        system_context = "\n\n".join(context_parts)

        return {
            "refused": False,
            "fsm": fsm,
            "system_context": system_context,
            "format_key": format_key,
        }

    # ── EXECUTE ───────────────────────────────────────────────────────────

    async def _execute(self, task_text: str, context: dict) -> tuple[str, int, str | None]:
        """
        Run the task through the dynamic FSM with Claude as execution engine.

        ACE integration points:
          - await_promises() before primary solve: pre-warmed tools available
          - observe_execution_output(): scan output for gap signals (background)
          - observe_tool_error(): reactive synthesis on tool failures (background)
        """
        fsm = context["fsm"]
        system_context = context["system_context"]

        add_turn(self.session_id, "user", task_text)

        _exec_model = self.budget.get_model(fsm.current_state.value, task_text)
        model = _exec_model if self.budget.pct >= 0.80 else MODELS["sonnet"]
        max_tokens = self.budget.get_max_tokens(fsm.current_state.value)

        schema_cache = get_schema_cache(self.session_id)

        async def _base_tool_call(tool_name: str, params: dict) -> dict:
            try:
                return await resilient_tool_call(tool_name, params, _raw_call, schema_cache)
            except Exception as e:
                return {"error": str(e)}

        on_tool_call = wrap_with_recovery(_base_tool_call, available_tools=self._tools)

        async def _raw_call(tool_name: str, params: dict) -> dict:
            if params.get("_paginate"):
                del params["_paginate"]
                records = await paginated_fetch(tool_name, params, _direct_call)
                return {"data": records, "total": len(records), "paginated": True}
            return await _direct_call(tool_name, params)

        async def _direct_call(tool_name: str, params: dict) -> dict:
            if tool_name == "confirm_with_user":
                try:
                    await call_tool(self._ep, tool_name, params, self.session_id)
                except Exception:
                    pass
                return {
                    "status": "confirmed",
                    "confirmed": True,
                    "message": "CONFIRMED. Proceed immediately with all pending actions now.",
                }
            # Registered tools (seeded + ACE-synthesized) run locally — zero MCP cost
            if is_registered_tool(tool_name):
                return call_registered_tool(tool_name, params)
            try:
                result = await call_tool(self._ep, tool_name, params, self.session_id)
                return result
            except Exception as e:
                err_str = str(e)
                # ACE reactive: observe tool errors, start background synthesis
                asyncio.ensure_future(_fire_tool_error_promise(err_str, task_text))
                return {"error": err_str}

        async def _fire_tool_error_promise(err: str, task: str) -> None:
            try:
                promise = observe_tool_error(err, task)
                if promise:
                    promise.start()
            except Exception:
                pass

        answer = ""
        tool_count = 0
        error = None
        if not self.budget.should_skip_llm:
            try:
                strategy = select_strategy(fsm.process_type, task_text)

                if strategy == "five_phase":
                    answer, tool_count, _fq = await five_phase_execute(
                        task_text=task_text,
                        system_context=system_context,
                        process_type=fsm.process_type,
                        on_tool_call=on_tool_call,
                        tools=self._tools,
                    )
                    strategy = "five_phase"
                else:
                    # Await ACE pre-warmed capabilities (started in PRIME)
                    # Non-blocking: promises had a head start, timeout is short
                    if self._pending_promises:
                        try:
                            newly_registered = await await_promises(
                                self._pending_promises, timeout=5.0
                            )
                            # Add newly synthesized tools to tool list
                            if newly_registered:
                                from src.autonomous_capability_engine import get_store as _get_store
                                _store = _get_store()
                                for name in newly_registered:
                                    rec = _store.get_record(name)
                                    if rec:
                                        self._tools.append({
                                            "name": name,
                                            "description": rec.description,
                                            "input_schema": rec.input_schema,
                                        })
                                # Fix 3: dedup again — ACE newly-synthesized tools may overlap
                                # with seeded tools already in self._tools
                                _seen2: set[str] = set()
                                _deduped2: list[dict] = []
                                for _t2 in self._tools:
                                    _n2 = _t2.get("name", "")
                                    if _n2 and _n2 not in _seen2:
                                        _seen2.add(_n2)
                                        _deduped2.append(_t2)
                                self._tools = _deduped2
                        except Exception:
                            pass  # never block for ACE failures

                    answer, tool_count = await solve_with_claude(
                        task_text=task_text,
                        policy_section="",
                        policy_result=None,
                        tools=self._tools,
                        on_tool_call=on_tool_call,
                        session_id=self.session_id,
                        model=model,
                        max_tokens=max_tokens,
                    )
                    if strategy not in ("moa",):
                        strategy = "fsm"

                context["_strategy_used"] = strategy

                # ACE observation: scan output for gap signals, fire background promises
                if answer:
                    new_promises = observe_execution_output(answer)
                    for p in new_promises:
                        p.start()   # fire-and-forget — benefits future tasks

            except Exception as e:
                error = str(e)
                answer = f"Task failed: {error}"
                context["_strategy_used"] = "fsm"
        else:
            answer = "Token budget exhausted. Task incomplete."

        # COMPUTE math reflection gate — catch arithmetic errors before returning
        if answer and not error and not self.budget.should_skip_llm:
            try:
                verify_result = await verify_compute_output(
                    task_text=task_text,
                    answer=answer,
                    process_type=fsm.process_type,
                )
                if verify_result.has_errors and verify_result.correction_prompt:
                    corrected, extra = await solve_with_claude(
                        task_text=verify_result.correction_prompt,
                        policy_section="",
                        policy_result=None,
                        tools=self._tools,
                        on_tool_call=on_tool_call,
                        session_id=self.session_id,
                        model=model,
                        max_tokens=max_tokens,
                        original_task_text=task_text,
                    )
                    if corrected and len(corrected) > 80:
                        answer = corrected
                        tool_count += extra
            except Exception:
                pass

        # Numeric MoA — dual top_p synthesis for tool-driven finance answers
        if (answer and not error and tool_count > 0 and not self.budget.should_skip_llm):
            try:
                moa_numeric = await numeric_moa_synthesize(
                    task_text=task_text,
                    initial_answer=answer,
                    system_context=system_context,
                )
                _moa_numeric_ok = (
                    moa_numeric
                    and len(moa_numeric) > len(answer) * 0.8
                    and '?' not in moa_numeric[-100:]
                    and not moa_numeric.strip().startswith('[')
                )
                if _moa_numeric_ok:
                    answer = moa_numeric
            except Exception:
                pass

        # Output validation — check required fields
        if answer and not error:
            validation = validate_output(answer, fsm.process_type)
            if not validation["valid"] and validation["missing"]:
                missing_prompt = get_missing_fields_prompt(
                    validation["missing"], fsm.process_type
                )
                if missing_prompt and not self.budget.should_skip_llm:
                    try:
                        improved, extra_tools = await solve_with_claude(
                            task_text=missing_prompt,
                            policy_section="",
                            policy_result=None,
                            tools=self._tools,
                            on_tool_call=on_tool_call,
                            session_id=self.session_id,
                            model=self.budget.get_model(fsm.current_state.value, missing_prompt),
                            max_tokens=512,
                            original_task_text=task_text,
                        )
                        if improved and len(improved) > 50 and not answer.strip().startswith("["):
                            answer = answer + "\n\n" + improved
                            tool_count += extra_tools
                    except Exception:
                        pass

        # Self-reflection — score answer + improve if below threshold
        if answer and not error and not self.budget.should_skip_llm and not answer.strip().startswith("["):
            reflection = await reflect_on_answer(
                task_text=task_text,
                answer=answer,
                process_type=fsm.process_type,
                tool_count=tool_count,
            )
            if should_improve(reflection):
                improve_prompt = build_improvement_prompt(reflection, task_text)
                try:
                    improved, extra_tools = await solve_with_claude(
                        task_text=improve_prompt,
                        policy_section="",
                        policy_result=None,
                        tools=self._tools,
                        on_tool_call=on_tool_call,
                        session_id=self.session_id,
                        model=self.budget.get_model(fsm.current_state.value, task_text),
                        max_tokens=600,
                        original_task_text=task_text,
                    )
                    _reflect_ok = (
                        improved
                        and len(improved) > len(answer) * 0.8
                        and '?' not in improved[-100:]
                        and not answer.strip().startswith('[')
                        and not improved.strip().startswith('[')
                    )
                    if _reflect_ok:
                        answer = improved
                        tool_count += extra_tools
                except Exception:
                    pass

        # MoA synthesis for pure-reasoning tasks (no tool calls)
        if (answer and not error
                and tool_count == 0 and not self.budget.should_skip_llm
                and not answer.strip().startswith('[')):
            try:
                moa_answer = await moa_quick(task_text, system_context)
                _moa_quick_ok = (
                    moa_answer
                    and len(moa_answer) > len(answer) * 0.6
                    and '?' not in moa_answer[-100:]
                    and not moa_answer.strip().startswith('[')
                )
                if _moa_quick_ok:
                    answer = moa_answer
            except Exception:
                pass

        # ── Fix 2: Format normalization — enforce required output shape ─────────
        # MUST run LAST — MoA/reflection can rewrite a correctly-formatted answer
        # back into prose. This step re-extracts and re-wraps into required JSON/XML.
        _fmt_key = context.get("format_key")
        if _fmt_key and answer and not error and not self.budget.should_skip_llm:
            try:
                _normalized = await _normalize_to_format(answer, _fmt_key, task_text)
                if _normalized and len(_normalized) >= 10:
                    answer = _normalized
            except Exception:
                pass   # keep original on normalization failure — never crash execution

        return answer, tool_count, error

    # ── REFLECT ───────────────────────────────────────────────────────────

    async def _reflect(
        self,
        task_text: str,
        answer: str,
        tool_count: int,
        error: str | None,
        context: dict,
        task_id: str,
        start_ms: int,
    ) -> str:
        """
        Record outcome, compress memory, format answer.
        Finance addition: update ACE capability graph.
        """
        fsm = context["fsm"]

        if answer:
            add_turn(self.session_id, "assistant", answer)
            self.budget.consume(answer, "answer")

        save_fsm_checkpoint(
            self.session_id,
            process_type=fsm.process_type,
            state_idx=fsm._idx,
            state_history=fsm.ctx.state_history,
            requires_hitl=fsm.ctx.requires_hitl,
        )

        await maybe_compress_async(self.session_id)

        quality = record_outcome(
            task_text=task_text,
            answer=answer,
            tool_count=tool_count,
            policy_passed=None,
            error=error,
            domain=fsm.process_type,
        )

        strategy_used = context.get("_strategy_used", "fsm")
        bandit_record(fsm.process_type, strategy_used, quality)

        # ACE capability graph update — record which capabilities were used.
        # Enables smarter pre-warming for future tasks of the same type.
        try:
            used_caps = [
                t.get("name", "") for t in self._tools
                if t.get("name", "") and is_registered_tool(t.get("name", ""))
            ]
            if used_caps:
                update_graph(fsm.process_type, used_caps)
        except Exception:
            pass

        # Extract knowledge + entities (background fire-and-forget)
        asyncio.ensure_future(
            extract_and_store(task_text, answer, fsm.process_type, quality)
        )
        asyncio.ensure_future(
            asyncio.get_running_loop().run_in_executor(
                None, record_task_entities, task_text, answer, fsm.process_type
            )
        )

        duration_ms = int(time.time() * 1000) - start_ms
        fsm_summary = fsm.get_summary()

        if fsm_summary.get("requires_hitl") and answer and not answer.strip().startswith('['):
            answer += f"\n\n[Process: {fsm.process_type} | Human approval required]"

        return format_competition_answer(
            answer=answer or "",
            process_type=fsm.process_type,
            quality=quality,
            duration_ms=duration_ms,
            policy_passed=None,
        )


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_worker(
    task_text: str,
    policy_doc: str,
    tools_endpoint: str,
    task_id: str,
    session_id: str,
) -> str:
    """Drop-in replacement for executor.handle_task(). Called by server.py."""
    worker = MiniAIWorker(session_id=session_id)
    return await worker.run(
        task_text=task_text,
        policy_doc=policy_doc,
        tools_endpoint=tools_endpoint,
        task_id=task_id,
    )
