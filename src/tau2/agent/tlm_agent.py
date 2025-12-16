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
    ToolMessage,
    UserMessage,
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

from cleanlab_tlm.utils.chat import (
    form_prompt_string,
    form_response_string_chat_completions,
)


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

        assistant_message: AssistantMessage = generate(  # type: ignore
            model=self.llm,
            tools=self.tools,
            messages=messages,
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
            query=message_content,  # type: ignore
            context=self.system_prompt,
            messages=chat_completion_messages,  # type: ignore
            tools=[tool.openai_schema for tool in self.tools],  # type: ignore
            metadata=(
                {"task_id": self.llm_args.get("task_id")}
                if self.llm_args.get("task_id")
                else {}
            ),
        )

        if assistant_message.raw_data is None:
            assistant_message.raw_data = {}

        groundedness = None

        if "response_groundedness" in validation_result.eval_scores:
            groundedness = validation_result.eval_scores["response_groundedness"].score

        assistant_message.raw_data["trustworthiness"] = groundedness

        if (
            "response_groundedness" in validation_result.eval_scores
            and validation_result.eval_scores["response_groundedness"].triggered
        ):
            find_reason_prompt = f"""Prior Messages:
{form_prompt_string(chat_completion_messages, [tool.openai_schema for tool in self.tools])}

Review the Response to the query and assess whether every factual claim in the Response is explicitly supported by the provided Context.
A Response meets the criteria if all information is directly backed by evidence in the Context, without relying on assumptions, external knowledge, or unstated inferences.
The focus is on whether the Response is fully grounded in the Context, rather than whether it fully addresses the query.
If any claim in the Response lacks direct support or introduces information not present in the Context, the Response is bad and does not meet the criteria.
Your job is to find out why the Response is not fully grounded in the Context.
You will now be provided with the Response and Context.
Just respond with the reason(s) why the Response is not fully grounded in the Context.

Context:
{self.system_prompt}

Response:
{assistant_message.content if assistant_message.content else ""}
{dumps([tc.model_dump() for tc in assistant_message.tool_calls], indent=2) if assistant_message.tool_calls else ""}"""
            reasons: AssistantMessage = generate(
                model=self.llm,
                tools=self.tools,
                messages=[
                    UserMessage(
                        role="user",
                        content=find_reason_prompt,
                    ),
                ],
                # **self.llm_args,
            )  # type: ignore

            revision_prompt = f"""Now, based on the reasons you just provided, revise the previous Response to ensure it is fully grounded in the Context.
Here is a reminder of the Response:
{assistant_message.content if assistant_message.content else ""}
{dumps([tc.model_dump() for tc in assistant_message.tool_calls], indent=2) if assistant_message.tool_calls else ""}

Only respond with the revised Response."""
            new_assistant_message: AssistantMessage = generate(
                model=self.llm,
                tools=self.tools,
                messages=[
                    UserMessage(
                        role="user",
                        content=find_reason_prompt,
                    ),
                    reasons,
                    UserMessage(
                        role="user",
                        content=revision_prompt,
                    ),
                ],
                # **self.llm_args,
            )  # type: ignore

            revised_validation = project.validate(
                response=message_to_chat_completion(new_assistant_message),
                query=message_content,  # type: ignore
                context=self.system_prompt,
                messages=chat_completion_messages,  # type: ignore
                tools=[tool.openai_schema for tool in self.tools],  # type: ignore
                metadata=(
                    {"task_id": self.llm_args.get("task_id")}
                    if self.llm_args.get("task_id")
                    else {}
                ),
            )

            if new_assistant_message.raw_data is None:
                new_assistant_message.raw_data = {}

            revised_groundedness = None

            if "response_groundedness" in revised_validation.eval_scores:
                revised_groundedness = revised_validation.eval_scores[
                    "response_groundedness"
                ].score

            assistant_message.raw_data["trustworthiness"] = revised_groundedness

            new_assistant_message.raw_data["trustworthiness"] = revised_groundedness

            if (
                revised_groundedness
                and revised_groundedness > groundedness  # type: ignore
            ):
                new_assistant_message.raw_data["previous_trustworthiness"] = (
                    groundedness
                )
                new_assistant_message.raw_data["previous_content"] = (
                    assistant_message.content
                )

                new_assistant_message.raw_data["previous_tool_calls"] = (
                    assistant_message.tool_calls
                )

                print("New assistant message is more grounded")
                print("Old message:", assistant_message.content)
                print("New message:", new_assistant_message.content)

                assistant_message = new_assistant_message
            else:
                assistant_message.raw_data["attempt_trustworthiness"] = (
                    revised_groundedness
                )
                assistant_message.raw_data["attempt_content"] = (
                    new_assistant_message.content
                )
                assistant_message.raw_data["attempt_tool_calls"] = (
                    new_assistant_message.tool_calls
                )

        state.messages.append(assistant_message)
        return assistant_message, state
