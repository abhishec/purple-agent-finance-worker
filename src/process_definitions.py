"""
process_definitions.py
Finance Mini AI Worker — fully dynamic process definitions.

PROCESS_DEFINITIONS is intentionally empty.
All process types are synthesized by Haiku at first encounter via dynamic_fsm.py.
One Haiku call per new type (~$0.0001), then cached indefinitely.

See: src/dynamic_fsm.py → synthesize_if_needed()
"""
from __future__ import annotations

PROCESS_DEFINITIONS: dict = {}
