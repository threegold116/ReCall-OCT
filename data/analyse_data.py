import json
import pandas as pd
import openai
from math_verify import parse, verify
from dateutil import parser
import re_search
import re_search_math

def has_digit(s):
    for ch in s:
        if ch.isdigit():  # 检查字符是否是数字
            return True
    return False
def is_date(s):
    try:
        parser.parse(s, fuzzy=False)   # 严格解析
        return True
    except (ValueError, OverflowError):
        return False
parquet_files = [
    "/share/home/sxjiang/myproject/ReCall-OCT/data/ReCall-data/syntool_re_call/train.parquet"
]
dataframes = []
for parquet_file in parquet_files:
    # read parquet files and cache
    dataframe = pd.read_parquet(parquet_file)
    dataframes.append(dataframe)
rl_data = pd.concat(dataframes)
math_item=0
qa_item=0
for idx in range(len(rl_data)):
    data_item = rl_data.iloc[idx].to_dict()
    # print(data_item.keys())
    answer = data_item["reward_model"]["ground_truth"]
    question = data_item["question"]
    # if len(answer)==1:
    #     print(answer)
    # if data_item["ability"]=="math":
    func_schemas_list = eval(data_item['extra_info']['func_schemas'].replace("null","None").replace("true","True").replace("false","False"))
    for idx in range(len(func_schemas_list)):
        import random
        cost = random.randint(1, 10)
        try:
            function_name = func_schemas_list[idx]['function']['name']
        except:
            function_name = func_schemas_list[idx]['name']
    
    if len(data_item["reward_model"]["ground_truth"])>1:
        pass
    ground_truth = answer[0]
    is_match = re_search_math.is_equiv(ground_truth,ground_truth)
    is_match_2 = re_search.em_check(ground_truth,ground_truth)
    if not is_match and not is_match_2:
        pass
    # if data_item["ability"]=="qa":
        # qa_item+=1
    # if "A." in question:
    #     print(question)
    #     print(answer)
        
print(f"math_item:{math_item}")
print(f"qa_item:{qa_item}")