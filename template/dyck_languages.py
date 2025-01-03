import json
from pydantic import BaseModel, constr


allowed_symbols = constr(pattern=r'^[<\[\]>{}]+$')

class ResponseType(BaseModel):
    response: allowed_symbols


data_path = "./dataset/bbh/dyck_languages.json"


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split("Input:")[1]
        if i <= (len_data // 2):
            train_set.append(([query], data["examples"][i]["target"]))
        else:
            test_set.append(([query], data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:str):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]}""" + """  Input: [ { [
A: ] } ]
""" + \
f"""
Q: {best_prompt[1]}""" + """  Input: < > ( ( [ [ ( { } ) [ < > ] ]
A: ] ) )
""" + \
f"""
Q: {best_prompt[1]}""" + """  Input: < [ < [ { < [ ] < { } > > } ] > { { ( ) } { < [ < > ] > }
A: } ] >
""" 
    usr_template = f"""Q: {best_prompt[1]}  Input: {usr_query[0]}
A:
"""
    return sys_template, usr_template


init_prompt = (
    "Correctly close a Dyck-n word.",
    "Complete the rest of the sequence, making sure that the parentheses are closed properly."
)

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
