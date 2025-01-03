import json
from pydantic import BaseModel
from enum import Enum


data_path = "./dataset/bbh/disambiguation_qa.json"

class SelectStr(str, Enum):
    a = "(A)"
    b = "(B)"
    c = "(C)"

class ResponseType(BaseModel):
    response: SelectStr


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split('\n')[1:]
        if i <= (len_data // 2):
            train_set.append((query, data["examples"][i]["target"]))
        else:
            test_set.append((query, data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:str):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]}
Sentence: {best_prompt[2]}
Options:
(A) The chief took the day off
(B) The counselor took the day off
(C) Ambiguous
A: (A)

Q: {best_prompt[1]}
Sentence: {best_prompt[3]}
Options:
(A) The secretary didn't reply yet
(B) The manager didn't reply yet
(C) Ambiguous
A: (A)

Q: {best_prompt[1]}
Sentence: {best_prompt[4]}
Options:
(A) It will be Bailey's office
(B) It will be the director's office
(C) Ambiguous
A: (C)

""" 
    usr_template = f"""Q: {best_prompt[1]}
Sentence: {usr_query[0]}
{usr_query[1]}
{usr_query[2]}
{usr_query[3]}
{usr_query[4]}
A:
"""
    return sys_template, usr_template


init_prompt = (
    "Clarify the meaning of sentences with ambiguous pronouns.",
    "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.",
    "The chief told the counselor that they took the day off.",
    """The manager sent a message to the secretary, but he didn't reply yet.""",
    """Bailey will plan to meet the director at his office."""
)

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
