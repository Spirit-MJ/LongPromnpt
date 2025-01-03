def mutation_prompt(sentences_to_rephrase: str, history_rewrite=None):
    new_history_rewrite = []
    if history_rewrite is not None:
        for x, y, score in history_rewrite:
            if score > 0: 
                new_history_rewrite.append((x, y))
            elif score < 0:
                new_history_rewrite.append((y, x))
            else:
                continue
    else:
        new_history_rewrite = history_rewrite
    rephrased_examples = "Here are some examples of how rephrasing a sentence can improve the performance of a language model:\n" + "".join(
    [f"<original>\n{rewrite[0]}\n</original>\n<rephrased>\n{rewrite[1]}\n</rephrased>\n" 
    for rewrite in new_history_rewrite]) if new_history_rewrite else ""
    sys_template  = f"""You are a professional sentence rephraser. Your job is to take a given sentence, and produce a new sentence that is easier for language models to understand.\n{rephrased_examples}
""" 
    
    usr_template = f"""
Rephrase the next sentence to enhance the language model's performance.
<original>\n{sentences_to_rephrase}\n</original>
<rephrased>\n
"""
    return sys_template, usr_template