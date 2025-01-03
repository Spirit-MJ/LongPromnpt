import json
from pydantic import BaseModel
from enum import Enum


data_path = "./dataset/bbh/logical_deduction_five_objects.json"

class ResponseStr(str, Enum):
    a = "(A)"
    b = "(B)"
    c = "(C)"
    d = "(D)"
    e = "(E)"

class ResponseType(BaseModel):
    response: ResponseStr


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split('The statements are logically consistent within each paragraph.')
        if i <= (len_data // 2):
            train_set.append((query, data["examples"][i]["target"]))
        else:
            test_set.append((query, data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:list):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]}  {best_prompt[2]}
In a golf tournament, there were three golfers: Amy, Eli, and Eve.    Eve finished above Amy.    Eli finished below Amy.
Options:
(A) Amy finished last
(B) Eli finished last
(C) Eve finished last
A: (B)

Q: {best_prompt[1]}  {best_prompt[2]}
On a shelf, there are three books: a white book, a green book, and an orange book.    The green book is to the right of the white book.    The orange book is the rightmost.
Options:
(A) The white book is the leftmost
(B) The green book is the leftmost
(C) The orange book is the leftmost
A: (A)

Q: {best_prompt[1]}  {best_prompt[2]}
On a shelf, there are three books: a red book, a gray book, and a white book.    The white book is to the left of the gray book.    The red book is the second from the left.
Options:
(A) The red book is the leftmost
(B) The gray book is the leftmost
(C) The white book is the leftmost
A: (C)

""" 
    usr_template = f"""Q:{best_prompt[3]}  {best_prompt[2]}
{usr_query[1]}
A:
"""
    return sys_template, usr_template


init_prompt = (
    "A logical deduction task which requires deducing the order of a sequence of objects.",
    "The following paragraphs each describe a set of three objects arranged in a fixed order.",
    "The statements are logically consistent within each paragraph.",
    "The following paragraphs each describe a set of five objects arranged in a fixed order."
   )

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
