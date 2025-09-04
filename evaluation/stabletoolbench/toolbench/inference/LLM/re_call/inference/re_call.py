from json import tool
import re
import json
import requests
import time
from typing import List
from functools import wraps
import random
#### load tool_cost dict
import os
cost_dict_path = os.path.join(os.path.dirname(__file__), "tool_cost.json")
print(cost_dict_path)
# if not os.path.exists(cost_dict_path):
#     os.makedirs(os.path.dirname(cost_dict_path), exist_ok=True)
#     with open(cost_dict_path, "w") as f:
#         json.dump({}, f)
#         cost_dict={}
with open(cost_dict_path, "r") as f:
    cost_dict = json.load(f)
#### load tool_cost dict


def remove_boxed(s):
    if s is None:
        return "no answer"
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval
def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] try {i} times")
                    if i == max - 1:
                        print(e)
                        raise Exception("Retry {} failed after {} times".format(func.__name__, max))
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

class ReCall():
    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    system_prompt_with_budget = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
You should make every function call count and obtain useful results. You have a total budget of {total_budget}. The costs of the function calls are as follows:\

{cost_sentence}\
    
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    def __init__(self, model_url, executor_url=None,task="syntool_re_call",observation_truncate=False,max_turn=10,max_step_length=512,method="no_budget_limit",budget=10):
        self.model_url = model_url
        self.executor_url = executor_url
        self.max_observation_length = 512
        self.task = task
        self.observation_truncate = observation_truncate
        self.max_turn = max_turn
        self.method = method
        self.max_step_length=max_step_length
        self.budget = budget
        self.achieve_max_budget_result = '{"error":"The cost of this API exceeds your remaining budget of [left_budget].","response":""}'
    def init_prompt(self, func_schemas, question):
        if not isinstance(func_schemas, str):
            func_schemas = json.dumps(func_schemas,indent=4)
        system_prompt = f"<|im_start|>system\n{self.system_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        func_schemas = eval(func_schemas.replace("null","None").replace("true","True").replace("false","False"))
        if "no-budget" not in self.method.lower():
            cost_sentence_template = "{function_name}: {cost}"
            cost_sentences = []
            for idx in range(len(func_schemas)):
                try:
                    function_name = func_schemas[idx]['function']['name']
                except:
                    function_name = func_schemas[idx]['name']
                cost_dict[function_name] = cost_dict[function_name]
                cost_sentences.append(cost_sentence_template.format(function_name=function_name, cost=cost_dict[function_name]))
            cost_sentence = "\n".join(cost_sentences)
            system_prompt = f"<|im_start|>system\n{self.system_prompt_with_budget.format(func_schemas=func_schemas,cost_sentence=cost_sentence,total_budget=self.budget)}<|im_end|>"
        else:
            for idx in range(len(func_schemas)):
                if func_schemas[idx]['function']['name'] not in cost_dict:
                    cost = random.randint(1, 10)
                    print("new cost")
                    print(func_schemas[idx]['function']['name'])
                    print(len(cost_dict))
                    cost_dict[func_schemas[idx]['function']['name']] = cost
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        #写入cost_dict
        # with open(cost_dict_path, "w") as f:
        #     json.dump(cost_dict, f, indent=4)
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"
    
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            for tool_call, result in zip(tool_calls, results):
                try:
                    result_json = json.loads(result)
                    if len(result_json["error"])>0:
                        result ="error: \n"+result_json["error"]
                    else:
                        if isinstance(result_json["response"],dict):
                            result ="result: \n"+json.dumps(result_json["response"],indent=4)
                        else:
                            result ="result: \n"+result_json["response"]
                except Exception as e:
                    result = result
            if self.task=="stabletoolbench" and self.observation_truncate: #TODO:利用tokenizer截断
                result = result[:self.max_observation_length]
            tool_response_str += f"<tool_response>{tool_call}\n {result}\n</tool_response>\n"
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            print(f"Parse tool call failed: {tool_call_str}")
            return f"Parse tool call failed: {e}"
    
    def execute_tool_calls(self, env: str, tool_calls: List[str],tool_call_times: dict,tool_call_error_times: dict,cost: int) -> List[str]:
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=3)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)
        def rapidapi_tool_call(env, call):
            action_name = call['name']
            action_input = call['arguments']
            response, status_code =  env._step(action_name=action_name, action_input=action_input)
            return response, status_code
        
        if self.task=="syntool_re_call":
            results = []
            for tool_call in tool_calls:
                result = exe_tool_call(env, tool_call)
                results.append(result)
        elif self.task=="stabletoolbench":
            results = []
            for tool_call in tool_calls:
                try:
                    tool_call = json.loads(tool_call)
                except Exception as e:
                    tool_call = tool_call.replace('{"}','{}')
                    try:
                        tool_call = json.loads(tool_call)
                    except Exception as e:
                        error_reason = f"Parse tool call failed: {e}"
                        result = f'{{"error":"{error_reason}","response":""}}'
                        results.append(result)
                        tool_call_error_times["parse_tool"] += 1
                        tool_call_times["parse_tool"] += 1
                        continue
                result,status_code = rapidapi_tool_call(env, tool_call)
                cost_single_tool_call = 0
                cost_single_tool_call_name = ""
                try:
                    tool_call_times[tool_call["name"]] += 1
                    cost_single_tool_call = cost_dict[tool_call["name"]]
                    cost_single_tool_call_name = tool_call["name"] #为了在超过budget时删除这个calling_times
                except KeyError as e:
                    tool_call_times["unknown_tool"] += 1
                    cost_single_tool_call = 10
                    cost_single_tool_call_name = "unknown_tool"
                    if status_code==0:#调用正确
                        if f"unexpected_tool_{tool_call['name']}" not in tool_call_times: #TODO:逻辑有问题，应该是tool不在tool_call_times里面
                            tool_call_times[f"unexpected_tool_{tool_call['name']}"] = 0
                        tool_call_times[f"unexpected_tool_{tool_call['name']}"] += 1
                        if tool_call["name"] not in cost_dict:
                            cost_dict[tool_call["name"]] = random.randint(1, 10)
                            print("new cost")
                            print(tool_call["name"])
                            print(len(cost_dict))
                        cost_single_tool_call_name = f"unexpected_tool_{tool_call['name']}"
                if status_code!=0:
                    try:
                        tool_call_error_times[tool_call["name"]] += 1
                    except KeyError as e:
                        tool_call_error_times["unknown_tool"] += 1
                if "no-budget" not in self.method.lower():
                    if cost+cost_single_tool_call<=self.budget:
                        results.append(result)
                        cost += cost_single_tool_call
                    else:
                        tool_call_times[cost_single_tool_call_name] -= 1
                        results.append(self.achieve_max_budget_result.replace("[left_budget]",str(self.budget-cost)))
                else:
                    results.append(result)
                
        return results,tool_call_times,tool_call_error_times,cost
    
    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
        
    @retry(max=5, sleep=1)
    def run(self, env, func_schemas, question):
        curr_prompt = self.init_prompt(func_schemas, question)
        input_prompt = curr_prompt
        output_answer = ""
        tool_call_times = {}
        tool_call_error_times = {}
        cost = 0
        if isinstance(func_schemas, str):
            func_schemas = json.loads(func_schemas)
        for func in func_schemas:
            tool_call_times[func["function"]["name"]] = 0
            tool_call_error_times[func["function"]["name"]] = 0
        tool_call_times["unknown_tool"] = 0
        tool_call_times["parse_tool"] = 0
        tool_call_error_times["unknown_tool"] = 0
        tool_call_error_times["parse_tool"] = 0
        for _ in range(self.max_turn):
            response = requests.post(
                f'{self.model_url}/v1/completions', 
                json={
                    "prompt": curr_prompt,
                    "model": "ReCall",
                    "temperature": 0.0,
                    "max_tokens": self.max_step_length
                }
            ).json()
            if "choices" not in response:
                print("over length")
                output_answer = "no answer"
                break
            curr_prompt = self.cat_assistant_response(curr_prompt, response["choices"][0]['text'])
            # print(f"response: {response['choices'][0]['text']}")
            tool_calls: List[str] = self.extract_tool_calls(response["choices"][0]['text'])
            if len(tool_calls) == 0:
                output_answer = remove_boxed(last_boxed_only_string(response["choices"][0]['text']))
                break

            results, tool_call_times, tool_call_error_times,cost = self.execute_tool_calls(env, tool_calls,tool_call_times,tool_call_error_times,cost)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)
        tool_call_cost = {}
        for key in tool_call_times.keys():
            if "unknown" in key:
                tool_call_cost[key] = 10
            elif "parse_tool" in key:
                tool_call_cost[key] = 10
            elif "unexpected" in key:
                tool_call_cost[key] = cost_dict[key.replace("unexpected_tool_","")]
            else:
                tool_call_cost[key] = cost_dict[key] #在统计cost时，先把unexcepted_tool作为unknown_tool处理(cost=10)
        result = {
            "Prompt": input_prompt,
            "Sequence": curr_prompt,
            "Output": output_answer,
            "query": question,
            "tools": func_schemas,
            "tool_call_times": tool_call_times,
            "tool_call_error_times": tool_call_error_times,
            "tool_call_cost": tool_call_cost,
        }
        #写入cost_dict
        # with open(cost_dict_path, "w") as f:
        #     json.dump(cost_dict, f, indent=4)
        return result