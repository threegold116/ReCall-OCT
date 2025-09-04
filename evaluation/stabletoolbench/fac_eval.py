import torch
import pandas as pd

import transformers
from transformers import pipeline
from accelerate import Accelerator

import json
from json import JSONDecodeError

from tqdm import tqdm
import argparse
import os
from judget_llm_server import llm_evaluate_equivalence_batch
import asyncio
PROMPT="""
Given a query and an answer provided by an AI agent, you now need to determine the answer_status of whether the well solved the query, i.e. whether the need of the query is satisfied. You need to output "Unsolved" or "Solved" and your reason. You must obey the following rules:

You should response "Solved" when:
    1. If the answer well provides the information needed by the query, then it is "Solved". The answer does not need to be perfect, and it only needs to make a genuine attempt to address the query.
	2.	Consider only Completeness:
	    •	The answer attempts to address every part of the query, regardless of whether the information provided is factually correct or accurate, unless there is a severe factual error.
	3.	For Multi-part Queries:
	    •	For queries with multiple parts, all parts must be addressed for the answer to be considered “Solved”.
	4.	Genuine Attempt :
	    •	The answer makes a genuine attempt to provide the requested information or perform the requested task for all parts of the query. This includes scenarios where the answer concludes that “nothing” is a reasonable response (e.g., when the requested information does not exist or is not available, or a possible answer of the query is nothing and the model answers nothing after reasonable attempts).

You should response "Unsolved" when:
    1.	Refusal, Apology, or Non-engagement:
	    •	The answer includes a refusal or apology (e.g., “I'm sorry, I can't help with that”).
	    •	The answer does not directly engage with or address the query in any way.
	2.	Multi-part Queries:
	    •	If the query has multiple parts and at least one part is not well addressed.
    3. Severe Factual Error:
        •   If the answer contains a severe factual error that significantly impacts the usefulness of the information provided.


Additional Guidelines:
    1. VERY IMPORTANT: DO NOT BE TOO HARSH. The model does not need to be perfect, and the answer does not need to be flawless. It only needs to make a genuine attempt to address the query.
    1. DO NOT evaluate factual accuracy or correctness of the information provided based on your knowledge. Assume that the information provided is accurate and focus solely on whether the answer attempts to address all parts of the query, unless there is a severe factual error that conficts common knowledge.
	2.	Focus on Final Answer: Only the final answer is provided and should be considered, disregarding any processes that were used to generate the answer. You only need to judge whether the information need is satisfied.
	3.	Answer Completion: The agent does not need to detail how it arrived at the answer, only that the answer itself is complete and attempts to address the query.

Here are some examples:

Query 1: I'm planning a movie night with my friends and we want to watch a horror film. Can you recommend some popular horror movies available on streaming platforms in the US? Also, provide the runtime and IMDb ratings for these movies.
Answer 1: Here are some popular horror movies available on streaming platforms in the US:\n\n1. Knives Out\n   - Runtime: 130 minutes\n   - IMDb Rating: 7.9/10\n   - Available on: Netflix, Prime Video, Hulu, Amazon Prime Video\n\n2. Jumanji: The Next Level\n   - Runtime: 110 minutes\n   - IMDb Rating: 6.7/10\n   - Available on: Hulu, Amazon Prime Video, Netflix\n\nPlease note that availability may vary depending on your location and streaming platform subscriptions. Enjoy your movie night!
Answer Status: Solved0304
Reason: The answer addressed all parts of subqueries by providing a list of popular horror movies available on streaming platforms in the US, along with their runtime and IMDb ratings. Whether the film is horror is a factual matter that does not to be checked.

Query 2: I'm a screenwriter looking for inspiration for my next project. Can you help me find a list of critically acclaimed movies available on streaming platforms in the US? Also, provide me with the streaming sources for the movie 'Citizen Kane' and the basic information about the cast, including their names and professions.
Answer 2: The movie 'Citizen Kane' is available on the following streaming platforms:
- HBO Max: [Watch on HBO Max](https://play.hbomax.com/feature/urn:hbo:feature:GXduU_gwwz-NvjAEAAAAC)
- Amazon Prime Video: [Watch on Amazon Prime Video](https://www.amazon.com/Citizen-Kane-Orson-Welles/dp/B000I9YLWG)
- AppleTV+: [Watch on AppleTV+](https://tv.apple.com/us/movie/citizen-kane/umc.cmc.21zj0v11gnqbbqrebieh0vpk7)
- Disney+: Available on Disney+
- Netflix: Available on Netflix
Answer Status: Unsolved
Reason: The answer only addressed the first and second part of the query by providing the streaming sources for the movie 'Citizen Kane' but did not provide information about the cast or a list of critically acclaimed movies available on streaming platforms in the US. The response was incomplete and did not fully address the query.

Query: 
{query}
Answer: 
{answer}

Now give your reason and answer status in the following format:

Answer Status
xxx (can only be "Solved" or "Unsolved")
Reason
xxx
"""


cate2num ={
    "G1_instruction": 163,
    "G1_category": 153,
    "G1_tool": 158,
    "G2_instruction": 106,
    "G2_category": 124,
    "G3_instruction": 61
}



def parse_args():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument("--evaluation_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resume_evaluation", action="store_true")

    return parser.parse_args()


def get_response(text):
    text = text.lower().strip()
    
    if "unsolved" in text:
        return 0
    elif "solved" in text:
        return 1
    else:
        return 0
def get_response(text):
    text = text.lower().strip()
    
    if "unsolved" in text:
        return 0
    elif "solved" in text:
        return 1
    else:
        return 0
def main(args):
    #读取数据
    print(args)
    print(args.evaluation_dir.split("/"))
    model_name = args.evaluation_dir.split("/")[-1]
    mode_name = args.evaluation_dir.split("/")[-2]
    category_name = args.evaluation_dir.split("/")[-3]
    print(category_name)
    output_path = os.path.join(args.output_dir,category_name,mode_name,model_name,"result.metric.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and args.resume_evaluation:
        print(f"output_path: {output_path} exists")
        with open(output_path, "r") as f:
            evaluate_data = json.load(f)
            evaluated_files = []
            evaluated_data = []
            for data_item in evaluate_data:
                if data_item["llm_judge_output"] =="Error" :
                    continue
                evaluated_files.append(data_item["file_name"])
                evaluated_data.append(data_item)
            if len(evaluated_data) == cate2num[category_name]:
                print(f"Already evaluated {category_name} of {model_name} in {mode_name}")
                return
            to_evaluate_data = []
    else:
        evaluated_files = []
        evaluated_data = []
        to_evaluate_data = []
    
    
    
    for file in os.listdir(args.evaluation_dir):
        if file.endswith(".json"):
            evaluation_path = os.path.join(args.evaluation_dir, file)
            with open(evaluation_path, "r") as f:
                data_item = json.load(f)
                data_item["file_name"] = file
                if file in evaluated_files:
                    continue
                to_evaluate_data.append(data_item)
    prompt_list = []
    for data_item in to_evaluate_data:
        query = data_item['query']
        answer = data_item["Output"]
        prompt_list.append(PROMPT.format(query=query, answer=answer))
    llm_results = []
    batch_size = 20
    for i in range(0, len(prompt_list), batch_size):
        batch_prompts = prompt_list[i:i+batch_size]
        batch_llm_results = asyncio.run(llm_evaluate_equivalence_batch(
            prompts=batch_prompts,
                extract_answer=False
            ))
        llm_results.extend(batch_llm_results)

    correct_num=0
    invalid_num=0
    all_calling_times=0
    all_calling_costs=0
    for idx,llm_judge in enumerate(llm_results):
        llm_judge_result = get_response(llm_judge)
        data_item = to_evaluate_data[idx]
        data_item["llm_judge"] = llm_judge_result
        data_item["llm_judge_output"] = llm_judge
        data_item["llm_judge_prompt"] = prompt_list[idx]
        if "no answer" in data_item["Output"].lower():
            invalid_num+=1
            
        tool_call_times = data_item["tool_call_times"]
        tool_call_cost = data_item["tool_call_cost"]
        for key,value in tool_call_times.items():
            all_calling_times+=value
            if "unexpected" in key:
                continue
            all_calling_costs+=value*tool_call_cost[key]                
        correct_num+=llm_judge_result
    with open(output_path, "w") as f:
        evaluated_data.extend(to_evaluate_data)
        json.dump(evaluated_data, f, indent=4)
        print(f"Saved results to {output_path}")
    llm_fac = correct_num/len(evaluated_data)
    overall_result ={
        "total_num": len(evaluated_data),
        "correct_num": correct_num,
        "invalid_num": invalid_num,
        "llm_fac": llm_fac,
        "all_calling_times": all_calling_times,
        "all_calling_costs": all_calling_costs
    }
    print(overall_result)
    with open(os.path.join(os.path.dirname(output_path), "result.overall.metric.json"), "w") as f:
       json.dump(overall_result, f, indent=4)
       print(f"Saved overall results to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
    