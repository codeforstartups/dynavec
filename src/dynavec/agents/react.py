"""ReAct (Reason + Act) tool-calling agent implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from ..chat.base import ChatModel, Message, Tool
from .base import AgentResult, AgentStep, AgentTool

DEFAULT_REACT_SYSTEM_PROMPT = """You are a helpful and precise reasoning agent.
You solve tasks step-by-step using a ReAct (Reason + Act) approach.
When you need to look up information or perform calculations, invoke the appropriate tools.
When you have the final answer to the user's goal, provide the clear and complete answer directly without calling any tools."""


class ReActAgent:
    """A ReAct-style agent that alternates between thinking, calling tools, and observing results."""

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[AgentTool | Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
    ) -> None:
        self.model = model
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_REACT_SYSTEM_PROMPT
        )
        self.max_steps = max(1, max_steps)

        self._tools_map: dict[str, AgentTool] = {}
        self._chat_tools: list[Tool] = []

        if tools:
            for t in tools:
                agent_tool = t if isinstance(t, AgentTool) else AgentTool(t)
                self._tools_map[agent_tool.name] = agent_tool
                self._chat_tools.append(agent_tool.to_chat_tool())

    def run(self, goal: str, **kwargs: Any) -> AgentResult:
        """Execute the ReAct loop synchronously until goal completion or max_steps."""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=goal))

        steps: list[AgentStep] = []
        total_tool_calls = 0

        for step_idx in range(1, self.max_steps + 1):
            chat_res = self.model.invoke(
                messages,
                tools=self._chat_tools if self._chat_tools else None,
                **kwargs,
            )
            msg = chat_res.message
            messages.append(msg)

            if not msg.tool_calls:
                # Final response reached without further tool calls
                step = AgentStep(
                    step_number=step_idx,
                    thought=msg.content,
                    tool_calls=[],
                    observations=[],
                )
                steps.append(step)
                return AgentResult(
                    output=msg.content or "",
                    steps=steps,
                    finished=True,
                    termination_reason="completed",
                    total_steps=step_idx,
                    tool_calls_count=total_tool_calls,
                )

            # Execute tool calls
            step_observations: list[str] = []
            for tc in msg.tool_calls:
                total_tool_calls += 1
                tool_instance = self._tools_map.get(tc.name)
                if tool_instance is not None:
                    obs = tool_instance.execute(tc.arguments)
                else:
                    obs = f"Error: Tool {tc.name!r} is not registered in available tools."

                step_observations.append(obs)
                messages.append(
                    Message(
                        role="tool",
                        content=obs,
                        tool_call_id=tc.id,
                    )
                )

            step = AgentStep(
                step_number=step_idx,
                thought=msg.content,
                tool_calls=list(msg.tool_calls),
                observations=step_observations,
            )
            steps.append(step)

        # Reached max steps without completing
        last_output = steps[-1].thought or (
            steps[-1].observations[-1] if steps[-1].observations else ""
        )
        return AgentResult(
            output=last_output,
            steps=steps,
            finished=False,
            termination_reason="max_steps_reached",
            total_steps=self.max_steps,
            tool_calls_count=total_tool_calls,
        )

    async def arun(self, goal: str, **kwargs: Any) -> AgentResult:
        """Execute the ReAct loop asynchronously."""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=goal))

        steps: list[AgentStep] = []
        total_tool_calls = 0

        for step_idx in range(1, self.max_steps + 1):
            chat_res = await self.model.ainvoke(
                messages,
                tools=self._chat_tools if self._chat_tools else None,
                **kwargs,
            )
            msg = chat_res.message
            messages.append(msg)

            if not msg.tool_calls:
                step = AgentStep(
                    step_number=step_idx,
                    thought=msg.content,
                    tool_calls=[],
                    observations=[],
                )
                steps.append(step)
                return AgentResult(
                    output=msg.content or "",
                    steps=steps,
                    finished=True,
                    termination_reason="completed",
                    total_steps=step_idx,
                    tool_calls_count=total_tool_calls,
                )

            step_observations: list[str] = []
            for tc in msg.tool_calls:
                total_tool_calls += 1
                tool_instance = self._tools_map.get(tc.name)
                if tool_instance is not None:
                    obs = await asyncio.to_thread(tool_instance.execute, tc.arguments)
                else:
                    obs = f"Error: Tool {tc.name!r} is not registered in available tools."

                step_observations.append(obs)
                messages.append(
                    Message(
                        role="tool",
                        content=obs,
                        tool_call_id=tc.id,
                    )
                )

            step = AgentStep(
                step_number=step_idx,
                thought=msg.content,
                tool_calls=list(msg.tool_calls),
                observations=step_observations,
            )
            steps.append(step)

        last_output = steps[-1].thought or (
            steps[-1].observations[-1] if steps[-1].observations else ""
        )
        return AgentResult(
            output=last_output,
            steps=steps,
            finished=False,
            termination_reason="max_steps_reached",
            total_steps=self.max_steps,
            tool_calls_count=total_tool_calls,
        )
