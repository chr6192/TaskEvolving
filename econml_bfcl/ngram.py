import re

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, min_scaling: float=0.0):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def preserve_chinese(text):
        # 使用正则表达式匹配所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 将匹配到的中文字符拼接成一个字符串
        return ''.join(chinese_chars)

    def zipngram_chinese(text: str, ngram_size: int):
        import jieba
        text = preserve_chinese(text)
        seg_list = list(jieba.cut(text))
        # print(seg_list)
        return zip(*[seg_list[i:] for i in range(ngram_size)])


    def repetition_penalty_reward(queries, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            queries: List of model queries
        """
        
        rewards = []
        for query in queries:

            ngrams = set()
            total = 0
            for ng in zipngram_chinese(query, ngram_size):
                ngrams.add(ng)
                total += 1

            if total == 0:
                rewards.append(0.0)
                continue

            # total 是定值（句子分词长度）
            # 假设 全部不重复 total = len(ngrams)，scaling ~= 0, reward=0
            # 假设 全部重复 len(ngrams) = 1, scaling ~= 1, reward=-1
            scaling = 1 - len(ngrams) / total
            # add soft penaly
            if scaling < min_scaling:
                scaling = 0.0
            elif scaling > min_scaling:
                scaling = (scaling-min_scaling)/(1-min_scaling)
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
