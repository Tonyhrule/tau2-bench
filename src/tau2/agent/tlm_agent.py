from copy import deepcopy
from typing import Any, List, Optional

from cleanlab_tlm import TLM

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
    APICompatibleMessage,
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate, to_litellm_messages
from cleanlab_tlm.utils.chat import (
    form_prompt_string,
    form_response_string_chat_completions_api,
)


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

    def trustworthiness_from_messages(
        self, messages: list[APICompatibleMessage], assistant_message: AssistantMessage
    ) -> Any:
        """Calculate the trustworthiness of the response based on the messages."""

        instruction_list = self.domain_policy.replace("### ", "").split("## ")[2:]
        instruction_dic = {}
        for instruction in instruction_list:
            instruction_name = instruction.split("\n")[0]
            instruction_dic[instruction_name] = instruction
        if "Book flight" in instruction_dic:
            # Airline
            instructions = {
                "book_reservation": instruction_dic["Book flight"],
                "update_reservation_baggages": instruction_dic["Modify flight"],
                "update_reservation_flights": instruction_dic["Modify flight"],
                "update_reservation_passengers": instruction_dic["Modify flight"],
                "cancel_reservation": instruction_dic["Cancel flight"],
                "send_certificate": instruction_dic["Refunds and Compensation"],
            }
        elif "Exchange delivered order" in instruction_dic:
            # Retail
            instructions = {
                "cancel_pending_order": instruction_dic["Cancel pending order"],
                "modify_pending_order_address": instruction_dic["Modify pending order"],
                "modify_pending_order_payment": instruction_dic["Modify pending order"],
                "modify_pending_order_items": instruction_dic["Modify pending order"],
                "return_delivered_order_items": instruction_dic[
                    "Return delivered order"
                ],
                "exchange_delivered_order_items": instruction_dic[
                    "Exchange delivered order"
                ],
            }
        elif "Data Refueling" in instruction_dic:
            # Telecom
            instructions = {
                "get_customer_by_phone": instruction_dic["Customer Lookup"],
                "get_customer_by_id": instruction_dic["Customer Lookup"],
                "get_customer_by_name": instruction_dic["Customer Lookup"],
                "send_payment_request": instruction_dic["Overdue Bill Payment"],
                "suspend_line": instruction_dic["Line Suspension"],
                "refuel_data": instruction_dic["Data Refueling"],
                "enable_roaming": instruction_dic["Data Roaming"],
                "disable_roaming": instruction_dic["Data Roaming"],
            }

        review_messages = deepcopy(messages)
        custom_eval_criteria = []

        if assistant_message.tool_calls:
            policy = ""

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.name
                if tool_name in instructions:
                    if instructions[tool_name] not in policy:
                        policy += f"\n\n## {instructions[tool_name]}"

            if "transfer_to_human_agents" in [
                tool_call.name for tool_call in assistant_message.tool_calls
            ]:
                custom_eval_criteria.append(
                    {
                        "name": "human_transfer",
                        "criteria": f"""Determine if the agent has exhausted all possibilities using the policy before transferring to human agents.
Policy:
{self.domain_policy}""",
                    }
                )
            else:
                custom_eval_criteria.append(
                    {
                        "name": "policy_compliance",
                        "criteria": f"""Determine if the response follows the domain policy: {policy}""",
                    }
                )

        tlm = TLM(
            options={
                "log": ["explanation"],
                "custom_eval_criteria": custom_eval_criteria,
            }
        )

        openai_messages = to_litellm_messages(review_messages)  # type: ignore
        for message in openai_messages:
            if "tool_calls" in message and message["tool_calls"] is None:
                del message["tool_calls"]

        openai_response = to_litellm_messages([assistant_message])[0]
        if "tool_calls" in openai_response and openai_response["tool_calls"] is None:
            del openai_response["tool_calls"]

        return tlm.get_trustworthiness_score(
            form_prompt_string(openai_messages, self.tools_info),
            form_response_string_chat_completions_api(openai_response),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message: AssistantMessage = generate(  # type: ignore
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )
        trustworthiness = self.trustworthiness_from_messages(
            messages, assistant_message
        )

        failed_custom_criteria = [
            criteria["name"]
            for criteria in trustworthiness["log"].get("custom_eval_criteria", [])
            if criteria["score"] < 0.75
        ]

        if assistant_message.raw_data is None:
            assistant_message.raw_data = {}

        assistant_message.raw_data["trustworthiness"] = trustworthiness

        if trustworthiness["trustworthiness_score"] < 0.75 or failed_custom_criteria:
            canceled_tool_messages = []
            if assistant_message.tool_calls:
                canceled_tool_messages = [
                    ToolMessage(
                        id=tool_call.id,
                        role="tool",
                        content="Tool Call Canceled",
                        requestor="assistant",
                    )
                    for tool_call in assistant_message.tool_calls
                ]

            reasons = []
            openai_messages = to_litellm_messages(messages)  # type: ignore
            for msg in openai_messages:
                if "tool_calls" in msg and msg["tool_calls"] is None:
                    del msg["tool_calls"]
            pretty_messages = form_prompt_string(openai_messages, self.tools_info)

            if trustworthiness["trustworthiness_score"] < 0.75:
                reasons.append(trustworthiness["log"]["explanation"])

            def get_reason(prompt: str) -> str:
                return generate(
                    model=self.llm,
                    messages=[
                        SystemMessage(
                            role="system",
                            content="You are to respond consisely with only what the user tells you to, nothing else.",
                        ),
                        UserMessage(
                            role="user",
                            content=f"""{prompt}

Policy:
{self.domain_policy}

Message History:
{pretty_messages}""",
                        ),
                    ],
                ).content.strip()  # type: ignore

            if "human_transfer" in failed_custom_criteria:
                reason = get_reason(
                    "The agent transferred to a human agent without exhausting all possibilities using the policy. Respond with something it could have tried before transferring."
                )
                reasons.append(
                    f"The agent transferred to human agents without exhausting all possibilities using the policy. It still could have tried: {reason}"  # type: ignore
                )

            if "policy_compliance" in failed_custom_criteria:
                reason = get_reason(
                    "The agent's response did not follow the domain policy. Respond with what it did not follow."
                )
                reasons.append(
                    f"The agent's response did not follow the domain policy. The policy it violated was: {reason}"
                )

            reasons = "\n\n".join(reasons)

            correction: AssistantMessage = generate(  # type: ignore
                model=self.llm,
                messages=[
                    SystemMessage(
                        role="system",
                        content="You are to respond consisely with only what the user tells you to, nothing else.",
                    ),
                    UserMessage(
                        role="user",
                        content=f"""In the following message chain, the assistant's message was not trustworthy. Pinpoint the specific issues with the assistant's response and explain what it should have done instead.
Message History:
{pretty_messages}{form_response_string_chat_completions_api(to_litellm_messages([assistant_message])[0])}

Potential Reasons:
{reasons}""",
                    ),
                ],
                **self.llm_args,
            )

            new_assistant_message: AssistantMessage = generate(  # type: ignore
                model=self.llm,
                tools=self.tools,
                messages=messages
                + [assistant_message]
                + canceled_tool_messages
                + [
                    SystemMessage(
                        role="system",
                        content=f"""Your last response was not trustworthy. Rewrite your response to be more trustworthy.
Information:
{correction.content}""",
                    )
                ],
                **self.llm_args,
            )

            new_trustworthiness = self.trustworthiness_from_messages(
                messages,
                new_assistant_message,
            )

            if min(
                [new_trustworthiness["trustworthiness_score"]]
                + [
                    criteria["score"]
                    for criteria in new_trustworthiness["log"].get(
                        "custom_eval_criteria", []
                    )
                ]
            ) > min(
                [trustworthiness["trustworthiness_score"]]
                + [
                    criteria["score"]
                    for criteria in trustworthiness["log"].get(
                        "custom_eval_criteria", []
                    )
                ]
            ):
                if new_assistant_message.raw_data is None:
                    new_assistant_message.raw_data = {}

                new_assistant_message.raw_data["trustworthiness"] = new_trustworthiness
                new_assistant_message.raw_data["previous_trustworthiness"] = (
                    trustworthiness
                )
                new_assistant_message.raw_data["previous_content"] = (
                    assistant_message.content
                )

                new_assistant_message.raw_data["previous_tool_calls"] = (
                    assistant_message.tool_calls
                )
                assistant_message = new_assistant_message
            else:
                assistant_message.raw_data["attempt_trustworthiness"] = (
                    new_trustworthiness
                )
                assistant_message.raw_data["attempt_content"] = (
                    new_assistant_message.content
                )
                assistant_message.raw_data["attempt_tool_calls"] = (
                    new_assistant_message.tool_calls
                )

        state.messages.append(assistant_message)
        return assistant_message, state
