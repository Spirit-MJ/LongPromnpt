import random
import numpy as np
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from param import args
import importlib
module = importlib.import_module(f"template.{args.task}")
get_data_set = getattr(module, 'get_data_set', None)
prediction_prompt = getattr(module, 'prediction_prompt', None)
init_prompt = getattr(module, 'init_prompt', None)
json_schema = getattr(module, 'json_schema', None)
from mutation_prompt import mutation_prompt

LLM_API = "http://172.16.0.96:8000/v1"
LLM_MODEL = "qwen2.5-7b"

# LLM_API = "http://111.13.172.45:8000/v1"
# LLM_MODEL = "Qwen72B"

LLM_KEY = "sk-abc123def0987111"

path_to_save = f"./results/{args.task}_results_7B_random.json"
    

class LLM:
    def __init__(self):
        self.json_schema = json_schema
        self.client = OpenAI(
                base_url = LLM_API,
                api_key=LLM_KEY
            ) 
        
    def get_response(self, sys_pmt, usr_query, temperature=0.7):
        response = self.client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": sys_pmt},
                  {"role": "user", "content": usr_query}],
        temperature=temperature,
        response_format={
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": self.json_schema},
            } if temperature == 0 else None
            )
        
        return response.choices[0].message.content

    
    def question_answer(self, data, best_prompt):
        query, answer = data
        sys_pmt, usr_pmt = prediction_prompt(best_prompt, query)
        response = self.get_response(sys_pmt, usr_pmt, temperature=0)
        res = json.loads(response)["response"]
        return res, answer
    
    def model_eval(self, data_set, best_prompt, max_workers=args.num_workers):
        correct_num = 0
        partial_question_answer = partial(self.question_answer, best_prompt=best_prompt)
        len_data_set = len(data_set)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in enumerate(executor.map(partial_question_answer, data_set)):
                print(f"\ritreation:{idx+1}/{len_data_set} "
                    f"[{'▓' * round((idx+1)/len_data_set*60)} "
                    f"{(idx+1)/len_data_set*100:2.0f}% "
                    f"{'-' * (60 - round((idx+1)/len_data_set*60))}] ", end="")
                if res[0] == res[1]:
                    correct_num += 1
        print()
        return correct_num / len_data_set
    

class Similarity:
    def __init__(self, history_topK=args.history_topK):
        self.topK = history_topK
    
    def get_best_history(self, history_sentences):
        len_history = len(history_sentences)
        if len_history == 0:
            return None
        if len(history_sentences[-1]) != 3:
            if len_history > 1:
                return sorted(history_sentences[:-1], key=lambda x: abs(x[2]), reverse=True)[:self.topK]
            else:
                return None


def save_results(results, history, path_to_save):
    file_output = {"prompt":[],"sentence_mutation":[]}
    for idx in range(len(results)):
        prompt_ls, train_score, test_score = results[idx]
        sys_pmt, _ = prediction_prompt(prompt_ls, prompt_ls)
        file_output["prompt"].append({"Epoch":idx,
                                      "prompt":sys_pmt,
                                      "train_set_score":train_score,
                                      "test_set_score":test_score})
        if idx < len(results)-1:
            file_output["sentence_mutation"].append({"Epoch":idx+1,
                                                    "before":history[idx][0],
                                                    "after":history[idx][1],
                                                    "score_change":history[idx][2]})
    with open(path_to_save, 'w') as file:
        json.dump(file_output, file, ensure_ascii=False, indent=4)
    print(f"Finish save result to {path_to_save}")


def beam_search_with_history_and_linucb(dataset, initial_prompt, iterations=50, k=args.search_window):
    """
    使用历史指导和 Lin-UCB 的束搜索优化长提示
    :param dataset: tuple （训练集, 测试集）
    :param initial_prompt: list 初始提示，包含若干句子
    :param iterations: 最大迭代次数
    :param k: 束的大小
    """
    train_data, test_data = dataset
    similarity = Similarity()
    llm = LLM()
    search_history = []
    history_mutation = [[] for _ in range(len(initial_prompt))]
    print("初始prompt得分计算")
    init_train_score = llm.model_eval(train_data, initial_prompt)
    init_test_score = llm.model_eval(test_data, initial_prompt)
    print(f"0 / {iterations} Train score:{init_train_score:.2f}, Test score:{init_test_score:.2f}")
    results = [(initial_prompt, init_train_score, init_test_score)]

    for iteration in range(iterations):
        # 从候选池中随机选择一个提示
        candidate_pool = sorted(results, key=lambda x: x[1], reverse=True)[:min(k, len(results))]
        selected_prompt = list(random.choice(candidate_pool)[0])

        # 随机选择句子进行变异
        sentence_idx, selected_sentence = random.choice(list(enumerate(initial_prompt)))
 
        # 更新历史记录中的stbefore
        search_history.append((selected_sentence,))
        # 更新历史记录中的embedding
        history_mutation[sentence_idx].append((selected_sentence,)) 

        # 选定句子并生成变体
        if iteration > 0:
            few_shot = similarity.get_best_history(history_mutation[sentence_idx])
            muta_sys_pmt, muta_usr_pmt = mutation_prompt(selected_sentence, few_shot)
        else:
            muta_sys_pmt, muta_usr_pmt = mutation_prompt(selected_sentence)

        mutated_sentence = llm.get_response(muta_sys_pmt, muta_usr_pmt)
        
        # 更新句子搜索历史
        search_history[-1] += (mutated_sentence.strip().strip('</rephrased>'),)
        history_mutation[sentence_idx][-1] +=(mutated_sentence.strip().strip('</rephrased>'),)

        # 替换句子生成新提示并评估
        selected_prompt[sentence_idx] = mutated_sentence
        new_train_score = llm.model_eval(train_data, selected_prompt)
        new_test_score = llm.model_eval(test_data, selected_prompt)

        # 更新
        search_history[-1] += (new_train_score-results[-1][1],)
        history_mutation[sentence_idx][-1] +=(new_train_score-results[-1][1],)
        
        results.append((tuple(selected_prompt), new_train_score, new_test_score))
        initial_prompt = results[-1][0]
        print(f"{iteration+1} / {iterations} Train score:{results[-1][1]:.2f}, Test score:{results[-1][-1]:.2f}")
    return results, search_history


if __name__ == "__main__":
    train_test_dataset = get_data_set()
    results, search_history = beam_search_with_history_and_linucb(train_test_dataset, init_prompt, iterations=args.epoches)
    save_results(results, search_history, path_to_save)
    # 输出结果
    print("最佳提示:")
    best_prompt = sorted(results, key=lambda x: x[1], reverse=True)[0]
    sys_pmt, _ = prediction_prompt(best_prompt[0], best_prompt[0][0])
    print(sys_pmt)
    print("最佳得分:")
    print("Train: ", best_prompt[1],"Test: ", best_prompt[2])
