import json
import os
import matplotlib.pyplot as plt
import numpy as np
def draw_multi_lines(x,y_list,labels,result_dir,name):
    x = x[:min(len(x),min([len(y) for y in y_list]))]
    y_list = [y[:len(x)] for y in y_list]
    plt.figure(figsize=(12, 8))  # 宽度=12，高度=5，单位是英寸
    colors = plt.cm.tab20(np.arange(len(y_list)) / len(y_list))
    for y,label,color in zip(y_list,labels,colors):
        plt.plot(x, y, marker='o', label=label, color=color)  # 画折线图并加点
    # labelLines(plt.gca().get_lines(), zorder=2.5) 4
    # plt.plot(x, [128 for i in x], linestyle='--', color = "gray",label="rollout batchsize")
    plt.title(f"Comparsion of {name}")
    plt.legend()
    # max_x = x[y.index(max(y))]
    # max_y = max(y)
    # print(max_x,max_y)
    # # 添加一条竖线
    # plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')

    # # 添加文字标注
    # plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)
    if os.path.exists(os.path.join(result_dir,f"{name}.png")):
        os.remove(os.path.join(result_dir,f"{name}.png"))
    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()
import sys
if len(sys.argv) >1:
    experiment_names = sys.argv[1:]
else:
    experiment_names = [
        "Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_new_f1_score_no_warm_up",
        "Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128-clip_ratio_0.28_one_epoch_tool_star_no_warm_up_new"
    ]
analyse_experiment_dir = "/share/home/sxjiang/myproject/ReCall-OCT/analyse"

experiment2results = {}
for experiment_name in experiment_names:
    analyse_experiment_path = os.path.join(analyse_experiment_dir, experiment_name)
    if not os.path.exists(analyse_experiment_path):
        raise f"the experiment {experiment_name} not exists in {analyse_experiment_dir}"
    with open(os.path.join(analyse_experiment_path,"rollout_step2metrics.json"),"r") as f:
        print(os.path.join(analyse_experiment_path,"rollout_step2metrics.json"))
        experiment2results[experiment_name] = json.load(f)
result_dir = "/share/home/sxjiang/myproject/ReCall-OCT/analyse/comparison"


comparison_key = "group_oct_zero_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        print(experiment_name)
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([128-rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_oct_no_zero_num")

comparison_key = "group_advantage_zero_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([128-rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_advantage_no_zero_num")

comparison_key = "all_calling_times"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"all_calling_times")

comparison_key = "group_multi_tool_calling_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_multi_tool_calling_num")

comparison_key = "all_calling_times"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key]/1024 for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"avg_all_calling_times")

comparison_key = "max_calling_times"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"max_calling_times")

comparison_key = "max_calling_times_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"max_calling_times_num")

comparison_key = "format_error_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"format_error_num")

comparison_key = "wrong_rollout_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"wrong_rollout_num")

comparison_key = "group_score_no_positive_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_score_no_positive_num")

comparison_key = "group_advantage_zero_num_all_one"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_advantage_zero_num_all_one")


comparison_key = "group_oct_zero_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([128 - rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_oct_num")

comparison_key = "group_zero_calling_num_positive"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_zero_calling_num_positive")

comparison_key = "group_oct_smooth_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_oct_smooth_num")
comparison_key = "group_oct_smooth_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key]/(128-rollout_step2metrics[str(i)]["group_oct_zero_num"]+0.00001) for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_oct_smooth_num_radio")

comparison_key = "group_zero_calling_num_positive"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_zero_calling_num_positive")



comparison_key = "group_zero_calling_num_positive"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_zero_calling_num_positive")

comparison_key = "max_length"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"max_length")


comparison_key = "group_multi_tool_calling_num"
y_list = []
labels = []
for experiment_name,rollout_step2metrics in experiment2results.items():
    if isinstance(rollout_step2metrics,list):
        pass
    x = [int(i) for i in rollout_step2metrics.keys()]
    x = sorted(x)
    y_list.append([rollout_step2metrics[str(i)][comparison_key] for i in x])
    labels.append(experiment_name.split("Qwen2.5-3B-Instruct-final_sft_edition10-52-grpo_debug-bz_128")[-1])
os.makedirs(result_dir,exist_ok=True)
draw_multi_lines(x,y_list,labels,result_dir,"group_multi_tool_calling_num")