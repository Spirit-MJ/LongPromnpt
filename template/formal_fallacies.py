import json
from pydantic import BaseModel
from enum import Enum


data_path = "./dataset/bbh/formal_fallacies.json"

class ResponseStr(str, Enum):
    valid = "valid"
    invalid = "invalid"

class ResponseType(BaseModel):
    response: ResponseStr


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split('Is the argument, given the explicitly stated premises, deductively valid or invalid?')
        if i <= (len_data // 2):
            train_set.append(([query[0]], data["examples"][i]["target"]))
        else:
            test_set.append(([query[0]], data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:list):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]} The following argument pertains to this question: {best_prompt[4]}
{best_prompt[3]}
Options:
- valid
- invalid
A: valid

Q: {best_prompt[1]} The following argument pertains to this question: {best_prompt[5]}
{best_prompt[3]}
Options:
- valid
- invalid
A: valid

Q: {best_prompt[2]} The following argument pertains to this question: {best_prompt[6]}
{best_prompt[3]}
Options:
- valid
- invalid
A: invalid

""" 
    usr_template = f"""Q:{usr_query[0]}
{best_prompt[3]}
Options:
- valid
- invalid
A:
"""
    return sys_template, usr_template


init_prompt = (
    "Distinguish deductively valid arguments from formal fallacies.",
    "It is not always easy to see who is related to whom -- and in which ways. ",
    "It is not always easy to grasp who is consuming which products.",
    "Is the argument, given the explicitly stated premises, deductively valid or invalid?",
    "To begin with, Lesley is a close friend of Fernando.   Moreover, being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy.   It follows that Lesley is a great-grandfather of Leroy.",
    """Whoever is not a great-grandfather of Clyde is a stepbrother of Brian.   Being an ancestor of Dana is sufficient for not being a great-grandfather of Clyde.   We may conclude: Everyone who is an ancestor of Dana is a stepbrother of Brian, too.""",
    """Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both.   No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and, in the same time, a loyal buyer of Caress soap.   It follows that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap."""
)

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
