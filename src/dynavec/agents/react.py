"""ReAct (Reason + Act) tool-calling agent implementation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..chat.base import ChatModel, Message, Tool, ToolCall
from ..exceptions import NodeInterrupt
from .base import AgentResult, AgentStep, AgentTool, resume_with
from .registry import ToolRegistry
from .state import AgentState

if TYPE_CHECKING:
    from ..checkpoint import BaseCheckpointer
    from ..memory import BaseMemory

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


def _pending_tool_calls(messages: list[Message]) -> list[ToolCall]:
    """Tool calls in the last assistant message that have no tool reply yet."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == "assistant":
            answered = {m.tool_call_id for m in messages[i + 1 :] if m.role == "tool"}
            return [tc for tc in msg.tool_calls if tc.id not in answered]
    return []


def _completed_step(messages: list[Message], step_number: int) -> AgentStep:
    """Build the AgentStep for the last assistant tool batch once all calls are answered."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == "assistant":
            return AgentStep(
                step_number=step_number,
                thought=msg.content,
                tool_calls=list(msg.tool_calls),
                observations=[m.content or "" for m in messages[i + 1 :] if m.role == "tool"],
            )
    return AgentStep(step_number=step_number)


class ReActAgent:
    """A ReAct-style agent that alternates between thinking, calling tools, and observing results."""

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[AgentTool | Callable[..., Any]] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 10,
        checkpointer: BaseCheckpointer | None = None,
        memory: BaseMemory | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_REACT_SYSTEM_PROMPT
        )
        self.max_steps = max(1, max_steps)
        self.checkpointer = checkpointer
        self.memory = memory

        self._tools_map: dict[str, AgentTool] = {}
        self._chat_tools: list[Tool] = []

        if isinstance(tools, ToolRegistry):
            for t in tools.list_tools():
                self._tools_map[t.name] = t
                self._chat_tools.append(t.to_chat_tool())
        elif tools:
            for raw_tool in tools:
                agent_tool = raw_tool if isinstance(raw_tool, AgentTool) else AgentTool(raw_tool)
                self._tools_map[agent_tool.name] = agent_tool
                self._chat_tools.append(agent_tool.to_chat_tool())

    def _init_messages(
        self,
        goal: str,
        state: AgentState | None = None,
    ) -> list[Message]:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        user_content = goal
        if state is not None and state.data:
            shared_context = json.dumps(state.data, default=str, sort_keys=True)
            user_content = f"{goal}\n\n" f"Shared workflow context (JSON):\n{shared_context}"

        messages.append(Message(role="user", content=user_content))
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

    def _process_tool_calls(self, tool_calls: list[ToolCall], messages: list[Message]) -> list[str]:
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

    # ------------------------------------------------------------------ memory
    def _new_turn(self, goal: str, thread_id: str | None, state: AgentState | None) -> list[Message]:
        """Messages for a new run: prior conversation from memory, then this goal."""
        fresh = self._init_messages(goal, state)
        if self.memory is None or thread_id is None:
            return fresh
        history = self.memory.get_messages(thread_id)
        # The system prompt is stored once, on the thread's first turn.
        return list(self.memory.append(thread_id, fresh if not history else fresh[-1:]))

    def _remember(self, thread_id: str | None, output: str) -> None:
        if self.memory is not None and thread_id is not None:
            self.memory.append(thread_id, Message(role="assistant", content=output))

    def _finalize_result(self, steps: list[AgentStep], total_tool_calls: int) -> AgentResult:
        last_output = (
            steps[-1].thought or (steps[-1].observations[-1] if steps[-1].observations else "")
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

    # ------------------------------------------------------------------ checkpoints
    def _save(
        self,
        thread_id: str | None,
        goal: str,
        messages: list[Message],
        steps: list[AgentStep],
        total_tool_calls: int,
        *,
        node_id: str,
        step: int,
        **extra: Any,
    ) -> None:
        if self.checkpointer is None or thread_id is None:
            return
        self.checkpointer.put(
            thread_id,
            state={
                "goal": goal,
                "messages": [_message_to_dict(m) for m in messages],
                "steps": [_step_to_dict(s) for s in steps],
                "total_tool_calls": total_tool_calls,
                **extra,
            },
            node_id=node_id,
            step=step,
        )

    def _restore(
        self, thread_id: str | None, resume_from: str | None
    ) -> tuple[list[Message], list[AgentStep], int, int, dict[str, Any] | None]:
        """Load (messages, steps, total_tool_calls, start_step, state) from a checkpoint."""
        if self.checkpointer is None or thread_id is None:
            return [], [], 0, 1, None
        existing_cp = self.checkpointer.get(thread_id, resume_from)
        if existing_cp is None:
            return [], [], 0, 1, None
        state = existing_cp.state
        messages: list[Message] = []
        steps: list[AgentStep] = []
        total_tool_calls = 0
        if state is not None:
            messages = [_dict_to_message(m) for m in state.get("messages", [])]
            steps = [_dict_to_step(s) for s in state.get("steps", [])]
            total_tool_calls = int(state.get("total_tool_calls", 0))
        return messages, steps, total_tool_calls, existing_cp.step + 1, state

    # ------------------------------------------------------------------ interrupts
    def _interrupted(
        self,
        exc: NodeInterrupt,
        goal: str,
        thread_id: str | None,
        messages: list[Message],
        steps: list[AgentStep],
        total_tool_calls: int,
        step_idx: int,
    ) -> AgentResult:
        self._save(
            thread_id,
            goal,
            messages,
            steps,
            total_tool_calls,
            node_id="interrupt",
            step=step_idx,
            finished=False,
            interrupted={"node_id": exc.node_id, "payload": exc.payload, "step": step_idx},
        )
        return AgentResult(
            output=f"Interrupted at {exc.node_id!r}: awaiting human input.",
            steps=list(steps),
            finished=False,
            termination_reason="interrupted",
            total_steps=step_idx,
            tool_calls_count=total_tool_calls,
            interrupt_payload={
                "thread_id": thread_id,
                "node_id": exc.node_id,
                "payload": exc.payload,
                "messages": list(messages),
                "steps": list(steps),
                "total_tool_calls": total_tool_calls,
                "start_step": step_idx,
            },
        )

    def _prepare_resume(
        self,
        messages: list[Message] | None,
        start_step: int | None,
        thread_id: str | None,
        steps: list[AgentStep] | None,
        total_tool_calls: int | None,
    ) -> tuple[str, list[Message], list[AgentStep], int, int, list[ToolCall]]:
        if messages is None:
            if self.checkpointer is None or thread_id is None:
                raise ValueError(
                    "resume() needs the interrupted run's messages, or a checkpointer and thread_id."
                )
            cp = self.checkpointer.get(thread_id)
            state = (cp.state if cp is not None else None) or {}
            paused = state.get("interrupted")
            if not paused:
                raise ValueError(f"Thread {thread_id!r} has no paused run to resume.")
            run_messages = [_dict_to_message(m) for m in state.get("messages", [])]
            run_steps = [_dict_to_step(s) for s in state.get("steps", [])]
            total = int(state.get("total_tool_calls", 0))
            start = int(paused["step"])
            goal = str(state.get("goal") or "")
        else:
            run_messages = list(messages)
            run_steps = list(steps or [])
            total = int(total_tool_calls or 0)
            start = start_step if start_step is not None else len(run_steps) + 1
            goal = next((m.content or "" for m in run_messages if m.role == "user"), "")

        pending = _pending_tool_calls(run_messages)
        if not pending:
            raise ValueError("Nothing to resume: no tool call is waiting for a human decision.")
        return goal, run_messages, run_steps, total, start, pending

    # ------------------------------------------------------------------ loops
    def _loop(
        self,
        goal: str,
        thread_id: str | None,
        messages: list[Message],
        steps: list[AgentStep],
        total_tool_calls: int,
        start_step: int,
        kwargs: dict[str, Any],
    ) -> AgentResult:
        for step_idx in range(start_step, self.max_steps + 1):
            chat_res = self.model.invoke(
                messages,
                tools=self._chat_tools if self._chat_tools else None,
                **kwargs,
            )
            msg = chat_res.message
            messages.append(msg)

            if not msg.tool_calls:
                steps.append(AgentStep(step_number=step_idx, thought=msg.content))
                self._remember(thread_id, msg.content or "")
                self._save(
                    thread_id, goal, messages, steps, total_tool_calls,
                    node_id=f"step_{step_idx}", step=step_idx,
                    output=msg.content or "", finished=True,
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
            try:
                step_observations = self._process_tool_calls(list(msg.tool_calls), messages)
            except NodeInterrupt as exc:
                return self._interrupted(
                    exc, goal, thread_id, messages, steps, total_tool_calls, step_idx
                )

            steps.append(
                AgentStep(
                    step_number=step_idx,
                    thought=msg.content,
                    tool_calls=list(msg.tool_calls),
                    observations=step_observations,
                )
            )
            self._save(
                thread_id, goal, messages, steps, total_tool_calls,
                node_id=f"step_{step_idx}", step=step_idx, finished=False,
            )

        res = self._finalize_result(steps, total_tool_calls)
        self._remember(thread_id, res.output)
        self._save(
            thread_id, goal, messages, steps, total_tool_calls,
            node_id="max_steps_reached", step=self.max_steps,
            output=res.output, finished=False,
        )
        return res

    async def _aloop(
        self,
        goal: str,
        thread_id: str | None,
        messages: list[Message],
        steps: list[AgentStep],
        total_tool_calls: int,
        start_step: int,
        kwargs: dict[str, Any],
    ) -> AgentResult:
        for step_idx in range(start_step, self.max_steps + 1):
            chat_res = await self.model.ainvoke(
                messages,
                tools=self._chat_tools if self._chat_tools else None,
                **kwargs,
            )
            msg = chat_res.message
            messages.append(msg)

            if not msg.tool_calls:
                steps.append(AgentStep(step_number=step_idx, thought=msg.content))
                self._remember(thread_id, msg.content or "")
                self._save(
                    thread_id, goal, messages, steps, total_tool_calls,
                    node_id=f"step_{step_idx}", step=step_idx,
                    output=msg.content or "", finished=True,
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
            try:
                step_observations = await self._aprocess_tool_calls(
                    list(msg.tool_calls), messages
                )
            except NodeInterrupt as exc:
                return self._interrupted(
                    exc, goal, thread_id, messages, steps, total_tool_calls, step_idx
                )

            steps.append(
                AgentStep(
                    step_number=step_idx,
                    thought=msg.content,
                    tool_calls=list(msg.tool_calls),
                    observations=step_observations,
                )
            )
            self._save(
                thread_id, goal, messages, steps, total_tool_calls,
                node_id=f"step_{step_idx}", step=step_idx, finished=False,
            )

        res = self._finalize_result(steps, total_tool_calls)
        self._remember(thread_id, res.output)
        self._save(
            thread_id, goal, messages, steps, total_tool_calls,
            node_id="max_steps_reached", step=self.max_steps,
            output=res.output, finished=False,
        )
        return res

    # ------------------------------------------------------------------ public API
    def _start(
        self, goal: str, thread_id: str | None, resume_from: str | None, state: AgentState | None
    ) -> tuple[list[Message], list[AgentStep], int, int]:
        messages, steps, total_tool_calls, start_step, cp_state = self._restore(
            thread_id, resume_from
        )
        if cp_state and cp_state.get("interrupted"):
            raise ValueError(
                f"Thread {thread_id!r} is paused waiting for human input; "
                "call resume(thread_id=..., human_decision=...) instead of run()."
            )
        if cp_state and cp_state.get("finished"):
            # A completed run is not resumable: a new run on the thread is a new turn.
            messages, steps, total_tool_calls, start_step = [], [], 0, 1
        if not messages:
            messages = self._new_turn(goal, thread_id, state)
            self._save(thread_id, goal, messages, [], 0, node_id="init", step=0)
        return messages, steps, total_tool_calls, start_step

    def run(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
        resume_from: str | None = None,
        state: AgentState | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the ReAct loop until it completes, hits max_steps, or a tool interrupts."""
        messages, steps, total, start = self._start(goal, thread_id, resume_from, state)
        return self._loop(goal, thread_id, messages, steps, total, start, kwargs)

    async def arun(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
        resume_from: str | None = None,
        state: AgentState | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the ReAct loop asynchronously."""
        messages, steps, total, start = self._start(goal, thread_id, resume_from, state)
        return await self._aloop(goal, thread_id, messages, steps, total, start, kwargs)

    def resume(
        self,
        messages: list[Message] | None = None,
        human_decision: Any = None,
        *,
        start_step: int | None = None,
        thread_id: str | None = None,
        steps: list[AgentStep] | None = None,
        total_tool_calls: int | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Continue an interrupted run with the human's decision.

        Pass the ``messages`` / ``start_step`` (and optionally ``steps``,
        ``total_tool_calls``) from ``AgentResult.interrupt_payload``, or just
        ``thread_id`` when the agent has a checkpointer. The paused tool is re-run and its
        :func:`~dynavec.agents.interrupt` call returns ``human_decision``.
        """
        goal, messages, steps, total, start, pending = self._prepare_resume(
            messages, start_step, thread_id, steps, total_tool_calls
        )
        try:
            with resume_with(human_decision):
                obs = self._dispatch_tool_call(pending[0])
            messages.append(Message(role="tool", content=obs, tool_call_id=pending[0].id))
            self._process_tool_calls(pending[1:], messages)
        except NodeInterrupt as exc:
            return self._interrupted(exc, goal, thread_id, messages, steps, total, start)

        steps.append(_completed_step(messages, start))
        self._save(
            thread_id, goal, messages, steps, total,
            node_id=f"step_{start}", step=start, finished=False,
        )
        return self._loop(goal, thread_id, messages, steps, total, start + 1, kwargs)

    async def aresume(
        self,
        messages: list[Message] | None = None,
        human_decision: Any = None,
        *,
        start_step: int | None = None,
        thread_id: str | None = None,
        steps: list[AgentStep] | None = None,
        total_tool_calls: int | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Asynchronous :meth:`resume`."""
        goal, messages, steps, total, start, pending = self._prepare_resume(
            messages, start_step, thread_id, steps, total_tool_calls
        )
        try:
            with resume_with(human_decision):
                obs = await self._adispatch_tool_call(pending[0])
            messages.append(Message(role="tool", content=obs, tool_call_id=pending[0].id))
            await self._aprocess_tool_calls(pending[1:], messages)
        except NodeInterrupt as exc:
            return self._interrupted(exc, goal, thread_id, messages, steps, total, start)

        steps.append(_completed_step(messages, start))
        self._save(
            thread_id, goal, messages, steps, total,
            node_id=f"step_{start}", step=start, finished=False,
        )
        return await self._aloop(goal, thread_id, messages, steps, total, start + 1, kwargs)
