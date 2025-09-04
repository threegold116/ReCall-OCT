import json
import os
from pyexpat import model

new_metrics = []
metric_dir="/home/sxjiang/myproject/agent/ReCall-OCT/evaluation/new_new_metrics"
cost_dict_path="/home/sxjiang/myproject/agent/ReCall-OCT/evaluation/stabletoolbench/toolbench/inference/LLM/re_call/inference/tool_cost.json"
for root,dirs,files in os.walk(metric_dir):
    for file in files:
        if file.endswith("result.metric.json"):
            # print(root, file)
            metric_path = os.path.join(root, file)
            if "156" in metric_path:
                continue
            with open(metric_path, "r") as f:
                metric_data = json.load(f)
            model_name = root.split("/")[-1]
            mode_name = root.split("/")[-2]
            category_name = root.split("/")[-3]
            total_cost = 0
            total_calling_times=0
            llm_fac_num=0
            with open(cost_dict_path, "r") as f:
                cost_dict = json.load(f)
            # print(len(cost_dict))
            for metric_item in metric_data:
                for tool_name,calling_times in metric_item["tool_call_times"].items():
                    if tool_name in cost_dict and calling_times>0:
                        total_cost += cost_dict[tool_name]*calling_times
                        total_calling_times += calling_times
                    elif tool_name in cost_dict and calling_times==0:
                        pass
                    elif tool_name =="unknown_tool" and calling_times>0:
                        total_calling_times += calling_times
                        total_cost += 10*calling_times #unknow_tool暂时设定为10，对于unexpected（不在dict但是调对了的，后期可以考虑加上其cost，然后unknow_tool-num）
                    elif tool_name =="unknown_tool" and calling_times==0:
                        pass
                    else:
                        # print(tool_name)
                        # print(calling_times)
                        import random
                        tool_name=tool_name.replace("unexpected_tool_","")
                        if tool_name in cost_dict:
                            continue
                        cost_dict[tool_name] = random.randint(1,10)
                        # print(metric_item["file_name"])
                        # print(cost_dict_path)
                        print(f"tool name {tool_name}")
                        with open(cost_dict_path, "w") as f:
                            json.dump(cost_dict, f, indent=4)
                        # print(f"tool name {tool_name}")
                        raise ValueError("no such tool")
                if metric_item["llm_judge_output"]=="Error":
                    print(metric_path)
                    break
                llm_fac_num+=metric_item["llm_judge"]
            new_metrics.append({
                "model_name": model_name,
                "mode_name": mode_name,
                "category_name": category_name,
                "total_num": len(metric_data),
                "llm_fac": 0 if len(metric_data)==0 else llm_fac_num/len(metric_data),
                "avg_calling_times": 0 if len(metric_data)==0 else total_calling_times/len(metric_data),
                "avg_calling_costs": 0 if len(metric_data)==0 else total_cost/len(metric_data),
            })

output_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(output_dir, "new_metrics.json"), "w") as f:
    json.dump(new_metrics, f, indent=4)
import pandas as pd
df = pd.read_json(os.path.join(output_dir, "new_metrics.json"))
df.to_csv(os.path.join(output_dir, "new_metrics.csv"), index=False)


