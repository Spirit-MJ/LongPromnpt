import json
from pydantic import BaseModel
from enum import Enum


data_path = "./dataset/bbh/salient_translation_error_detection.json"

class ResponseStr(str, Enum):
    a = "(A)"
    b = "(B)"
    c = "(C)"
    d = "(D)"
    e = "(E)"
    f = "(F)"

class ResponseType(BaseModel):
    response: ResponseStr


def get_data_set():
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    train_set, test_set = [], []  
    len_data = len(data["examples"])
    for i in range(len_data):
        query = data["examples"][i]["input"].split('Please identify that error. ')
        if i <= (len_data // 2):
            train_set.append((query, data["examples"][i]["target"]))
        else:
            test_set.append((query, data["examples"][i]["target"]))
    print(f"Finish loading dataset from {data_path}")
    return train_set, test_set



def prediction_prompt(best_prompt:list, usr_query:list):
    sys_template = f"""{best_prompt[0]}

Q: {best_prompt[1]}  {best_prompt[2]}  {best_prompt[3]}  {best_prompt[4]}  {best_prompt[5]}  {best_prompt[6]}  {best_prompt[7]}  {best_prompt[8]}
Source: In der Liste der Baudenkmale in Lenzen (Elbe) sind alle Baudenkmale der brandenburgischen Stadt Lenzen (Elbe) und ihrer Ortsteile aufgelistet.
Translation: In the list of architectural monuments in Lenzen all architectural monuments of the Brandenburg city of Lenzen and its districts are listed.
The translation contains an error pertaining to
Options:
(A) Modifiers or Adjectives
(B) Numerical Values
(C) Negation or Antonyms
(D) Named Entities
(E) Dropped Content
(F) Facts
A: (D)

Q: {best_prompt[1]}  {best_prompt[2]}  {best_prompt[3]}  {best_prompt[4]}  {best_prompt[5]}  {best_prompt[6]}  {best_prompt[7]}  {best_prompt[8]}
Source: Auf dieser Seite sind die Baudenkm\u00e4ler der oberbayerischen Gro\u00dfen Kreisstadt Landsberg am Lech zusammengestellt.
Translation: On this page are compiled the architectural monuments of the town of Landsberg am Lech.
The translation contains an error pertaining to
Options:
(A) Modifiers or Adjectives
(B) Numerical Values
(C) Negation or Antonyms
(D) Named Entities
(E) Dropped Content
(F) Facts
A: (E)

Q: {best_prompt[1]}  {best_prompt[2]}  {best_prompt[3]}  {best_prompt[4]}  {best_prompt[5]}  {best_prompt[6]}  {best_prompt[7]}  {best_prompt[8]}
Source: \u0141eba ist eine Kleinstadt und ein Badeort im Powiat L\u0119borski der polnischen Woiwodschaft Pommern.    
Translation: Eba is not a small town and seaside resort in the Powiat L\u00e9borski county of the Pomeranian Voivodeship of Poland.
The translation contains an error pertaining to
Options:
(A) Modifiers or Adjectives
(B) Numerical Values
(C) Negation or Antonyms
(D) Named Entities
(E) Dropped Content
(F) Facts
A: (C)

""" 
    usr_template = f"""Q: {best_prompt[1]}  {best_prompt[2]}  {best_prompt[3]}  {best_prompt[4]}  {best_prompt[5]}  {best_prompt[6]}  {best_prompt[7]}  {best_prompt[8]}
{usr_query[1]}
A:
"""
    return sys_template, usr_template


init_prompt = (
    "Detect the type of error in an English translation of a German source sentence.",
    "The following translations from German to English contain a particular error.     That error will be one of the following types: ",
    "Named Entities: An entity (names, places, locations, etc.) is changed to a different entity. ",
    "Numerical Values: Numerical values (ordinals or cardinals), dates, and/or units are changed. ",
    "Modifiers or Adjectives: The modifiers and adjectives pertaining to a noun are changed.",
    "Negation or Antonyms: Introduce or remove a negation or change comparatives to their antonyms.",
    "Facts: Trivial factual errors not pertaining to the above classes are introduced in the translations.",
    "Dropped Content: A significant clause in the translation is removed. ",
    "Please identify that error."
   )

json_schema = ResponseType.model_json_schema()


if __name__ == '__main__':
    train_data, test_data = get_data_set()
    sys_pmt, usr_pmt = prediction_prompt(init_prompt, train_data[0][0])
    print(sys_pmt, usr_pmt)
