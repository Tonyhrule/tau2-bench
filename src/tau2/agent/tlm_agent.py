from copy import deepcopy
from json import dumps, loads
import os
from typing import Any, List, Optional

from codex import Client

from tau2.agent.base import (
    ValidAgentInputMessage,
)
from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.guidance import consult
from tau2.utils.llm_utils import generate, to_litellm_messages

import uuid
import time

from cleanlab_codex import Project

from openai.types.chat import ChatCompletion

client = Client(api_key=os.getenv("CODEX_API_KEY"))
project = Project(client, os.getenv("CLEANLAB_PROJECT_ID", ""))

# from cleanlab_tlm.utils.chat import (
#     form_prompt_string,
#     form_response_string_chat_completions_`api`,
# )


def message_to_chat_completion(
    msg: AssistantMessage, model="dummy-model-v1"
) -> ChatCompletion:
    """
    Convert your internal message schema into a valid ChatCompletion API object.
    """

    message = loads(dumps(msg.model_dump()))

    if message.get("tool_calls") is not None:
        toolCalls = message.get("tool_calls")
        for tc in toolCalls:
            tc["function"] = {
                "name": tc["name"],
                "arguments": dumps(tc["arguments"]),
            }
            tc["type"] = "function"
            del tc["name"]
            del tc["arguments"]
    else:
        toolCalls = None

    usage = message.get("usage", {}) or {}

    # AUTO-FILL usage if missing
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    # BUILD ChatCompletion FORMAT
    chatCompletion = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": message.get("content"),
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    # If tool_calls exist → embed them inside choices[].message
    if toolCalls is not None:
        chatCompletion["choices"][0]["message"]["tool_calls"] = toolCalls

    return chatCompletion  # type: ignore


class TLMAgent(LLMAgent):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the TLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.tools_info = [tool.openai_schema for tool in tools] if tools else None

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            message_content = (
                f"Tool calls received: "
                f"{', '.join([str(tm) for tm in message.tool_messages])}."
            )
        else:
            state.messages.append(message)
            message_content = message.content
        messages = state.system_messages + state.messages

        guidance = consult(message_content, to_litellm_messages(messages))  # type: ignore

        assistant_message: AssistantMessage = generate(  # type: ignore
            model=self.llm,
            tools=self.tools,
            messages=messages
            + (
                [
                    SystemMessage(
                        role="system",
                        content="Remember the following:\n" + "\n".join(guidance),
                    )
                ]
                if guidance
                else []
            ),
            # **self.llm_args,
        )

        chat_completion_messages = to_litellm_messages(messages)  # type: ignore

        for msg in chat_completion_messages:
            if msg.get("arguments") is not None:
                msg["function"] = {
                    "name": msg["name"],
                    "arguments": dumps(msg["arguments"]),
                }
                msg["type"] = "function"
                del msg["name"]
                del msg["arguments"]
            if msg.get("tool_calls", "") is None:
                del msg["tool_calls"]

        validation_result = project.validate(
            response=message_to_chat_completion(assistant_message),
            query=message_content,
            context=self.system_prompt,
            messages=chat_completion_messages,
            tools=[tool.openai_schema for tool in self.tools],
            metadata=(
                {"task_id": self.llm_args.get("task_id")}
                if self.llm_args.get("task_id")
                else {}
            ),
        )

        print(validation_result)

        state.messages.append(assistant_message)
        return assistant_message, state
