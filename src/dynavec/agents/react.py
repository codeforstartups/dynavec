"""ReAct (Reason + Act) tool-calling agent implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..chat.base import ChatModel, Message, Tool, ToolCall
from .base import AgentResult, AgentStep, AgentTool
from .registry import ToolRegistry

if TYPE_CHECKING:
    from ..checkpoint import BaseCheckpointer

DEFAULT_REACT_SYSTEM_PROMPT = """You are a helpful and precise reasoning agent.
You solve tasks step-by-step using a ReAct (Reason + Act) approach.
When you need to look up information or perform calculations, invoke the appropriate tools.
When you have the final answer to the user's goal, provide the clear and complete answer directly without calling any tools."""


def _message_to_dict(msg: Message) -> dict[str, Any]:
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_call_id": msg.tool_call_id,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls
        ],
    }


def _dict_to_message(d: dict[str, Any]) -> Message:
    tcs = [
        ToolCall(id=t["id"], name=t["name"], arguments=t["arguments"])
        for t in d.get("tool_calls", [])
    ]
    return Message(
        role=d["role"],
        content=d.get("content"),
        tool_calls=tcs,
        tool_call_id=d.get("tool_call_id"),
    )


def _step_to_dict(step: AgentStep) -> dict[str, Any]:
    return {
        "step_number": step.step_number,
        "thought": step.thought,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in step.tool_calls
        ],
        "observations": list(step.observations),
    }


def _dict_to_step(d: dict[str, Any]) -> AgentStep:
    tcs = [
        ToolCall(id=t["id"], name=t["name"], arguments=t["arguments"])
        for t in d.get("tool_calls", [])
    ]
    return AgentStep(
        step_number=int(d["step_number"]),
        thought=d.get("thought"),
        tool_calls=tcs,
        observations=list(d.get("observations") or []),
    )


class ReActAgent:
    """A ReAct-style agent that alternates between thinking, calling tools, and observing results."""

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[AgentTool | Callable[..., Any]] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        checkpointer: BaseCheckpointer | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_REACT_SYSTEM_PROMPT
        )
        self.max_steps = max(1, max_steps)
        self.checkpointer = checkpointer

        self._tools_map: dict[str, AgentTool] = {}
        self._chat_tools: list[Tool] = []

        if isinstance(tools, ToolRegistry):
            for t in tools.list_tools():
                self._tools_map[t.name] = t
                self._chat_tools.append(t.to_chat_tool())
        elif tools:
            for raw_tool in tools:
                agent_tool = (
                    raw_tool if isinstance(raw_tool, AgentTool) else AgentTool(raw_tool)
                )
                self._tools_map[agent_tool.name] = agent_tool
                self._chat_tools.append(agent_tool.to_chat_tool())

    def _init_messages(self, goal: str) -> list[Message]:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=goal))
        return messages

    def _dispatch_tool_call(self, tc: ToolCall) -> str:
        tool_instance = self._tools_map.get(tc.name)
        if tool_instance is not None:
            return tool_instance.execute(tc.arguments)
        return f"Error: Tool {tc.name!r} is not registered in available tools."

    async def _adispatch_tool_call(self, tc: ToolCall) -> str:
        tool_instance = self._tools_map.get(tc.name)
        if tool_instance is not None:
            return await tool_instance.aexecute(tc.arguments)
        return f"Error: Tool {tc.name!r} is not registered in available tools."

    def _process_tool_calls(
        self, tool_calls: list[ToolCall], messages: list[Message]
    ) -> list[str]:
        step_observations: list[str] = []
        for tc in tool_calls:
            obs = self._dispatch_tool_call(tc)
            step_observations.append(obs)
            messages.append(
                Message(
                    role="tool",
                    content=obs,
                    tool_call_id=tc.id,
                )
            )
        return step_observations

    async def _aprocess_tool_calls(
        self, tool_calls: list[ToolCall], messages: list[Message]
    ) -> list[str]:
        step_observations: list[str] = []
        for tc in tool_calls:
            obs = await self._adispatch_tool_call(tc)
            step_observations.append(obs)
            messages.append(
                Message(
                    role="tool",
                    content=obs,
                    tool_call_id=tc.id,
                )
            )
        return step_observations

    def _finalize_result(
        self, steps: list[AgentStep], total_tool_calls: int
    ) -> AgentResult:
        last_output = (
            steps[-1].thought
            or (steps[-1].observations[-1] if steps[-1].observations else "")
            if steps
            else ""
        )
        return AgentResult(
            output=last_output,
            steps=steps,
            finished=False,
            termination_reason="max_steps_reached",
            total_steps=self.max_steps,
            tool_calls_count=total_tool_calls,
        )

    def run(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
        resume_from: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the ReAct loop synchronously until goal completion or max_steps."""
        messages: list[Message] = []
        steps: list[AgentStep] = []
        total_tool_calls = 0
        start_step = 1

        if self.checkpointer is not None and thread_id is not None:
            existing_cp = self.checkpointer.get(thread_id, resume_from)
            if existing_cp is not None:
                state = existing_cp.state
                if "messages" in state:
                    messages = [_dict_to_message(m) for m in state["messages"]]
                if "steps" in state:
                    steps = [_dict_to_step(s) for s in state["steps"]]
                total_tool_calls = int(state.get("total_tool_calls", 0))
                start_step = existing_cp.step + 1

        if not messages:
            messages = self._init_messages(goal)
            if self.checkpointer is not None and thread_id is not None:
                self.checkpointer.put(
                    thread_id,
                    state={
                        "goal": goal,
                        "messages": [_message_to_dict(m) for m in messages],
                        "steps": [],
                        "total_tool_calls": 0,
                    },
                    node_id="init",
                    step=0,
                )

        for step_idx in range(start_step, self.max_steps + 1):
            chat_res = self.model.invoke(
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
                if self.checkpointer is not None and thread_id is not None:
                    self.checkpointer.put(
                        thread_id,
                        state={
                            "goal": goal,
                            "messages": [_message_to_dict(m) for m in messages],
                            "steps": [_step_to_dict(s) for s in steps],
                            "total_tool_calls": total_tool_calls,
                            "output": msg.content or "",
                            "finished": True,
                        },
                        node_id=f"step_{step_idx}",
                        step=step_idx,
                    )
                return AgentResult(
                    output=msg.content or "",
                    steps=steps,
                    finished=True,
                    termination_reason="completed",
                    total_steps=step_idx,
                    tool_calls_count=total_tool_calls,
                )

            total_tool_calls += len(msg.tool_calls)
            step_observations = self._process_tool_calls(list(msg.tool_calls), messages)

            step = AgentStep(
                step_number=step_idx,
                thought=msg.content,
                tool_calls=list(msg.tool_calls),
                observations=step_observations,
            )
            steps.append(step)

            if self.checkpointer is not None and thread_id is not None:
                self.checkpointer.put(
                    thread_id,
                    state={
                        "goal": goal,
                        "messages": [_message_to_dict(m) for m in messages],
                        "steps": [_step_to_dict(s) for s in steps],
                        "total_tool_calls": total_tool_calls,
                        "finished": False,
                    },
                    node_id=f"step_{step_idx}",
                    step=step_idx,
                )

        res = self._finalize_result(steps, total_tool_calls)
        if self.checkpointer is not None and thread_id is not None:
            self.checkpointer.put(
                thread_id,
                state={
                    "goal": goal,
                    "messages": [_message_to_dict(m) for m in messages],
                    "steps": [_step_to_dict(s) for s in steps],
                    "total_tool_calls": total_tool_calls,
                    "output": res.output,
                    "finished": False,
                },
                node_id="max_steps_reached",
                step=self.max_steps,
            )
        return res

    async def arun(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
        resume_from: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the ReAct loop asynchronously."""
        messages: list[Message] = []
        steps: list[AgentStep] = []
        total_tool_calls = 0
        start_step = 1

        if self.checkpointer is not None and thread_id is not None:
            existing_cp = self.checkpointer.get(thread_id, resume_from)
            if existing_cp is not None:
                state = existing_cp.state
                if "messages" in state:
                    messages = [_dict_to_message(m) for m in state["messages"]]
                if "steps" in state:
                    steps = [_dict_to_step(s) for s in state["steps"]]
                total_tool_calls = int(state.get("total_tool_calls", 0))
                start_step = existing_cp.step + 1

        if not messages:
            messages = self._init_messages(goal)
            if self.checkpointer is not None and thread_id is not None:
                self.checkpointer.put(
                    thread_id,
                    state={
                        "goal": goal,
                        "messages": [_message_to_dict(m) for m in messages],
                        "steps": [],
                        "total_tool_calls": 0,
                    },
                    node_id="init",
                    step=0,
                )

        for step_idx in range(start_step, self.max_steps + 1):
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
                if self.checkpointer is not None and thread_id is not None:
                    self.checkpointer.put(
                        thread_id,
                        state={
                            "goal": goal,
                            "messages": [_message_to_dict(m) for m in messages],
                            "steps": [_step_to_dict(s) for s in steps],
                            "total_tool_calls": total_tool_calls,
                            "output": msg.content or "",
                            "finished": True,
                        },
                        node_id=f"step_{step_idx}",
                        step=step_idx,
                    )
                return AgentResult(
                    output=msg.content or "",
                    steps=steps,
                    finished=True,
                    termination_reason="completed",
                    total_steps=step_idx,
                    tool_calls_count=total_tool_calls,
                )

            total_tool_calls += len(msg.tool_calls)
            step_observations = await self._aprocess_tool_calls(
                list(msg.tool_calls), messages
            )

            step = AgentStep(
                step_number=step_idx,
                thought=msg.content,
                tool_calls=list(msg.tool_calls),
                observations=step_observations,
            )
            steps.append(step)

            if self.checkpointer is not None and thread_id is not None:
                self.checkpointer.put(
                    thread_id,
                    state={
                        "goal": goal,
                        "messages": [_message_to_dict(m) for m in messages],
                        "steps": [_step_to_dict(s) for s in steps],
                        "total_tool_calls": total_tool_calls,
                        "finished": False,
                    },
                    node_id=f"step_{step_idx}",
                    step=step_idx,
                )

        res = self._finalize_result(steps, total_tool_calls)
        if self.checkpointer is not None and thread_id is not None:
            self.checkpointer.put(
                thread_id,
                state={
                    "goal": goal,
                    "messages": [_message_to_dict(m) for m in messages],
                    "steps": [_step_to_dict(s) for s in steps],
                    "total_tool_calls": total_tool_calls,
                    "output": res.output,
                    "finished": False,
                },
                node_id="max_steps_reached",
                step=self.max_steps,
            )
        return res
