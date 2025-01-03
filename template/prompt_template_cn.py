def mutation_prompt(sentences_to_rephrase: str, history_rewrite=None):
    new_history_rewrite = [(x, y) if score >= 0 else (y, x) for x, y, score in history_rewrite] if history_rewrite is not None else history_rewrite 
    rephrased_examples = "以下是一些改写句子如何提高语言模型性能的例子：\n" + "".join(
    [f"<original>\n{rewrite[0]}\n</original>\n<rephrased>\n{rewrite[1]}\n</rephrased>\n" 
    for rewrite in new_history_rewrite]) if new_history_rewrite else ""
    sys_template  = f"""你是一个专业的句子改写者，你的工作是取一个给定的句子，并编写一个语言模型更容易理解的新句子。\n{rephrased_examples}
""" 
    
    usr_template = f"""
重新表述下边的句子，以提高语言模型的性能。
<original>\n{sentences_to_rephrase}\n</original>
<rephrased>\n
"""
    return sys_template, usr_template


def prediction_prompt(best_prompt:list, usr_query:str):
    sys_template = f"""
<task>
{best_prompt[0]}
</task>

<examples>
输入：地弹门M1034 单门-地弹+锁 单位:套。
输出：地弹门

输入：方形球磨铸铁溢流口 750*450*25mm 单位:套。
输出：方形球磨铸铁溢流口

输入：（详效果图）仿古水刷石外墙、仿古红砖外墙
输出：仿古水刷石外墙,仿古红砖外墙
</examples>

<instructions>
{best_prompt[1]}
</instructions>
""" 
    usr_template = f"""
用户的输入如下：<inputs>{usr_query[0]}</inputs>
"""
    return sys_template, usr_template


init_prompt = (
    "本任务要求从给定的文本输入中识别并提取出物料的名称。输出中不应包含任何额外的内容，不应该包含你的思考。",
    """1. 首先，分析输入文本，识别出所有可能的物料名称。
2. 如果有多个物料名称，请用','隔开，之后将物料名称输出。
3. 确保输出只包含物料名称，不包含属性规格等其余额外的内容。
4. **绝对禁止输出任何解释、分析、理由、额外说明或思考过程**。"""
)


if __name__ == '__main__':
    print(len(init_prompt))
    print(init_prompt[1])
