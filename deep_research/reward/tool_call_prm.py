
# todo：for tool call format validation and prm in the process of multi turn rollout
def tool_call_format_reward():
    """
    Check tool call format and calculate reward.

    Args:
        sample: Data sample containing generated content

    Returns:
        RewardResult: Reward result containing format score
    """
    content = sample.output[0].answer.content

    # Extract tag contents
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

    think_matches = re.search(think_pattern, content, re.DOTALL)
    answer_matches = re.search(answer_pattern, content, re.DOTALL)
    tool_call_matches = re.findall(tool_call_pattern, content, re.DOTALL)

    has_think_tag = think_matches is not None
    has_answer_tag = answer_matches is not None
    has_tool_call_tag = len(tool_call_matches) > 0

    valid_format = False
    valid_tool_call_json = False
    reasons = []

    if has_think_tag:
        # Case 1: <think></think> + <answer></answer>
        if has_answer_tag and not has_tool_call_tag:
            # Check overall format
            format_pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            valid_format = bool(re.match(format_pattern, content, re.DOTALL))

            # Check tag occurrence count
            if valid_format:
                valid_format = (
                        content.count("<think>") == 1
                        and content.count("</think>") == 1
                        and content.count("<answer>") == 1
                        and content.count("</answer>") == 1
                )

            if valid_format:
                reasons.append("Valid <think></think> + <answer></answer> format")
            else:
                reasons.append("Invalid <think></think> + <answer></answer> format")

        # Case 2: <think></think> + <tool_call></tool_call>
        elif has_tool_call_tag and not has_answer_tag:
            # Check overall format
            format_pattern = (
                r"^\s*<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>\s*)+$"
            )
            valid_format = bool(re.match(format_pattern, content, re.DOTALL))

            # Check <think> tag occurrence count
            if valid_format:
                valid_format = (
                        content.count("<think>") == 1 and content.count("</think>") == 1
                )

            # Check if <tool_call> and </tool_call> tags appear in pairs
            if valid_format:
                if content.count("<tool_call>") != content.count("</tool_call>"):
                    valid_format = False

            # Check for consecutive duplicate tags
            if valid_format:
                if re.search(r"</tool_call>\s*</tool_call>", content) or re.search(
                        r"<tool_call>\s*<tool_call>", content
                ):
                    valid_format = False

            # Check tool_call JSON format
            valid_tool_call_json = True
            tool_calls = []
            if valid_format:
                for tool_call_content in tool_call_matches:
                    try:
                        tool_call_json = json.loads(tool_call_content.strip())
                        # Check if JSON contains required fields
                        if not (
                                "name" in tool_call_json
                                and "arguments" in tool_call_json
                        ):
                            valid_tool_call_json = False
                            break
                        tool_calls.append(
                            {
                                "function": {
                                    "name": tool_call_json["name"],
                                    "arguments": json.dumps(
                                        tool_call_json["arguments"],
                                        ensure_ascii=False,
                                    ),
                                }
                            }
                        )
                    except json.JSONDecodeError:
                        valid_tool_call_json = False
                        break

            valid_format = valid_format and valid_tool_call_json

            if valid_format:
                reasons.append(
                    "Valid <think></think> + <tool_call></tool_call> format with valid JSON"
                )
            else:
                if not valid_tool_call_json:
                    reasons.append("Invalid JSON format in <tool_call> tags")
                else:
                    reasons.append(
                        "Invalid <think></think> + <tool_call></tool_call> format"
                    )
        else:
            # Has both answer and tool_call, or neither
            reasons.append(
                "Invalid combination: should have either <answer> or <tool_call> tags, not both or neither"
            )
    else:
        reasons.append("Missing <think></think> tags")

    # Calculate reward score
    reward = 1.0 if valid_format else 0.0

    return RewardResult(
        name=self.name,
        details=[
            RewardDimensionWithScore(
                name=self.name, score=reward, reason="; ".join(reasons)
            )
        ],
        extra_data={
            "has_think_tag": has_think_tag,
            "has_answer_tag": has_answer_tag,
            "has_tool_call_tag": has_tool_call_tag,
            "valid_format": valid_format,
            "valid_tool_call_json": valid_tool_call_json,
            "tool_call_count": len(tool_call_matches),
            "reward": reward,
        },
    )