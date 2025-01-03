import json
from pydantic import BaseModel
from enum import Enum


data_path = "./dataset/bbh/hyperbaton.json"

class ResponseStr(str, Enum):
    a = "(A)"
    b = "(B)"

class ResponseType(BaseModel):
    response: ResponseStr


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split('Which sentence has the correct adjective order:')
        if i <= (len_data // 2):
            train_set.append((query, data["examples"][i]["target"]))
        else:
            test_set.append((query, data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:list):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]}
Options:
(A) rubber terrible ship
(B) terrible rubber ship
A: (B)

Q: {best_prompt[1]}
Options:
(A) repulsive small Brazilian exercise ship
(B) Brazilian repulsive exercise small ship
A: (A)

Q: {best_prompt[1]}
Options:
(A) blue gold wonderful square shoe
(B) wonderful square blue gold shoe
A: (B)

""" 
    usr_template = f"""Q:{best_prompt[1]}
{usr_query[1]}
A:
"""
    return sys_template, usr_template


init_prompt = (
    "Order adjectives correctly in English sentences.",
    "Which sentence has the correct adjective order:",
   )

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
