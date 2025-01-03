import json
from pydantic import BaseModel
from enum import Enum


data_path = "./dataset/bbh/causal_judgement.json"

class BoolStr(str, Enum):
    yes = "Yes"
    no = "No"

class ResponseType(BaseModel):
    response: BoolStr


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split('\n')
        if i <= (len_data // 2):
            train_set.append(([query[1]], data["examples"][i]["target"]))
        else:
            test_set.append(([query[1]], data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:list):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]}
{best_prompt[2]}
Options:
- Yes
- No
A: No

Q: {best_prompt[1]}
{best_prompt[3]}
Options:
- Yes
- No
A: Yes

Q: {best_prompt[1]}
{best_prompt[4]}
Options:
- Yes
- No
A: Yes

""" 
    usr_template = f"""Q:{best_prompt[1]}
{usr_query[0]}
Options:
- Yes
- No
A:
"""
    return sys_template, usr_template


init_prompt = (
    "Answer questions about causal attribution.",
    "How would a typical person answer each of the following questions about causation?",
    "Frank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor's body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body?",
    """Suzy and Billy are working on a project that is very important for our nation's security. The boss tells them both: "Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time." Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off. Did Billy cause the motion detector to go off?""",
    """George and his sister Lena reunite at their parents' house for Thanksgiving. Whereas George just got into medical school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon they play a game of darts. They split the first two games, and the third game is close until the end. Who will win comes down to George's last shot. If he hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks of the difficult time Lena is having, and he really wants to let her win. He aims the dart at the low point region. He sets up his shot and the dart lands in the low point region. After his shot, Lena wins the game and is very happy. Did George hit the low point region intentionally?"""
)

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
