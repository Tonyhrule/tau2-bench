from json import load


PATH = "data/simulations/retail_commenting.json"

with open(PATH, "r") as file:
    data = load(file)

simulations = data["simulations"]

min_trustworthiness = [
    min(
        [
            min(
                [
                    (
                        message["raw_data"]["trustworthiness"]["trustworthiness_score"]
                        if "trustworthiness" in (message.get("raw_data", {}) or {})
                        and "trustworthiness_score"
                        in message["raw_data"]["trustworthiness"]
                        else 1
                    )
                ]
                + [
                    custom_criterion["score"]
                    for custom_criterion in (
                        message["raw_data"]["trustworthiness"]["log"][
                            "custom_eval_criteria"
                        ]
                        if "trustworthiness" in (message.get("raw_data", {}) or {})
                        and "log" in message["raw_data"]["trustworthiness"]
                        and "custom_eval_criteria"
                        in message["raw_data"]["trustworthiness"]["log"]
                        else []
                    )
                ]
            )
            for message in simulation["messages"]
        ]
    )
    for simulation in simulations
]

rewards = [simulation["reward_info"]["reward"] for simulation in simulations]


def get_accuracy(threshold: float) -> float:
    filtered_rewards = [
        reward
        for reward, trust in zip(rewards, min_trustworthiness)
        if trust >= threshold
    ]
    print(len(filtered_rewards))
    return sum(filtered_rewards) / len(filtered_rewards) if filtered_rewards else 0.0


for i in [x * 0.05 for x in range(0, 21)]:
    print(i, get_accuracy(i))
