"""Planner primitive for decomposing goals into structured action plans."""

from __future__ import annotations

import json
import re
from typing import Any

from ..chat.base import ChatModel, Message
from .base import Plan, PlanStep

DEFAULT_PLANNER_SYSTEM_PROMPT = """You are an expert task planning agent.
Your job is to break down complex goals into an ordered, clear sequence of actionable steps.
Output your plan strictly as a JSON object matching this schema:
{
  "steps": [
    {
      "step_number": 1,
      "description": "Description of what needs to be done in this step",
      "tool_hint": "Optional name of the tool or action suited for this step"
    }
  ]
}
Do not include any conversational filler or markdown other than the valid JSON."""


def _extract_json_block(text: str) -> str:
    """Extract JSON content from markdown code fences or plain text."""
    trimmed = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return trimmed


class Planner:
    """Decomposes goals into structured, ordered execution plans."""

    def __init__(
        self,
        model: ChatModel,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_PLANNER_SYSTEM_PROMPT
        )

    def _parse_plan(self, goal: str, response_text: str) -> Plan:
        """Parse raw model output into a Plan object."""
        cleaned = _extract_json_block(response_text)
        try:
            data = json.loads(cleaned)
        except Exception:
            # Fallback: if json parsing fails, split lines into steps
            lines = [
                line.strip()
                for line in response_text.splitlines()
                if line.strip() and not line.startswith("```")
            ]
            steps = [
                PlanStep(step_number=idx, description=line)
                for idx, line in enumerate(lines, start=1)
            ]
            return Plan(goal=goal, steps=steps)

        steps_data = data.get("steps", []) if isinstance(data, dict) else data
        plan_steps: list[PlanStep] = []

        if isinstance(steps_data, list):
            for idx, item in enumerate(steps_data, start=1):
                if isinstance(item, dict):
                    step_num = item.get("step_number", idx)
                    desc = item.get("description", str(item))
                    tool_hint = item.get("tool_hint")
                    plan_steps.append(
                        PlanStep(
                            step_number=step_num,
                            description=desc,
                            tool_hint=tool_hint,
                        )
                    )
                elif isinstance(item, str):
                    plan_steps.append(PlanStep(step_number=idx, description=item))

        return Plan(goal=goal, steps=plan_steps)

    def plan(
        self,
        goal: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> Plan:
        """Decompose a goal into an ordered Plan synchronously."""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        user_content = f"Goal: {goal}"
        if context:
            user_content += f"\n\nContext:\n{context}"
        messages.append(Message(role="user", content=user_content))

        res = self.model.invoke(messages, **kwargs)
        return self._parse_plan(goal, res.message.content or "")

    async def aplan(
        self,
        goal: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> Plan:
        """Decompose a goal into an ordered Plan asynchronously."""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        user_content = f"Goal: {goal}"
        if context:
            user_content += f"\n\nContext:\n{context}"
        messages.append(Message(role="user", content=user_content))

        res = await self.model.ainvoke(messages, **kwargs)
        return self._parse_plan(goal, res.message.content or "")
