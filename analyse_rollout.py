from cv2 import calibrateCamera
import jsonlines
import os
import json
import numpy as np
import re
from labellines import labelLines
import matplotlib.pyplot as plt
# from analyse_research import get_search_questions
import sys
if len(sys.argv) > 1:
    expertment_name = sys.argv[1]
else:
    expertment_name="Qwen2.5-3B-Instruct-zero-52-grpo-oct-epoch1"
data_dir = f"/share/home/sxjiang/myproject/ReCall-OCT/verl_checkpoints/{expertment_name}/rollout"
specific_rollout_iter_num = -1
questions_worng_dict = {}

def get_question(sentence):
    sentence = sentence.split("<|im_end|>\n<|im_start|>assistant\n")[0]
    sentence = sentence.split("<|im_start|>user\n")[1]
    return sentence
def find_wrong(reson_str,sequences_str):
    
    # # if "<think> </think> not paired" in reson_str:
    # #     return False
    # if sequences_str.count("</answer>")>3 or sequences_str.count("<answer>")>3:
    #     return True
    if "</think><result>" in sequences_str.replace(" ",""):
        return True
    # if "< <" in sequences_str:
    #     return True
    # if sequences_str.count("</answer>")==2:
    #     return False
    # if sequences_str.count("</answer>")==2 and sequences_str.count("<|im_end|>")==3:
    #     return False
    # if sequences_str.count("</answer>")!=sequences_str.count("<|im_end|>"):
    #     return True
    # if not sequences_str.strip().endswith("<|im_end|>") and not sequences_str.strip().endswith("</python>") and not sequences_str.strip().endswith("</search>"):
    #     print(sequences_str[-10:])
    #     return True
    # if sequences_str.strip().endswith("</python><|im_end|>") or sequences_str.strip().endswith("</search><|im_end|>"):
    #     return True
    return False
def draw_with_max(x,y,result_dir,name):
    plt.figure(figsize=(12, 5))  # 宽度=12，高度=5，单位是英寸
    plt.plot(x, y, marker='o', label='Line')  # 画折线图并加点
    max_x = x[y.index(max(y))]
    max_y = max(y)
    print(max_x,max_y)
    # 添加一条竖线
    plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')

    # 添加文字标注
    plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()
def draw_multi_lines(x,y_list,labels,result_dir,name):
    plt.figure(figsize=(12, 5))  # 宽度=12，高度=5，单位是英寸
    colors = plt.cm.tab20(np.arange(len(y_list)) / len(y_list))
    for y,label,color in zip(y_list,labels,colors):
        plt.plot(x, y, marker='o', label=label, color=color)  # 画折线图并加点
    # labelLines(plt.gca().get_lines(), zorder=2.5) 4
    plt.plot(x, [128 for i in x], linestyle='--', color = "gray",label="rollout batchsize")
    plt.legend()
    # max_x = x[y.index(max(y))]
    # max_y = max(y)
    # print(max_x,max_y)
    # # 添加一条竖线
    # plt.axvline(x=max_x, color='red', linestyle='--', label='Max Value')

    # # 添加文字标注
    # plt.text(max_x, max_y + 1, f'Max: {max_x}', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(result_dir,f"{name}.png"))
    plt.close()

rollout_step2metrics={}
wrong_rollout_idx = []
# print(data_dir)
wrong_rollout_num_dict={}
for rollout_file in os.listdir(data_dir): #TODO:增加对smooth的统计
    rollout_step = rollout_file.split("_")[1].split(".")[0]
    score_sum = 0
    max_calling_times = 0
    all_calling_times = 0
    all_observation_length = 0
    max_length = 0
    count = 0
    questions = []
    wrong_rollout_num=0
    wrong_rollout_sequences=[]
    format_error_num=0
    duplicate_rollout_num=0
    multi_tool_calling_num=0
    # print(rollout_file)
    if "val" in rollout_file:
        continue
    if str(specific_rollout_iter_num) not in rollout_file and specific_rollout_iter_num != -1:
        continue
    # print(os.path.join(data_dir, rollout_file))
    # with jsonlines.open(os.path.join(data_dir, rollout_file),encoding="utf-8") as reader:
    with open(os.path.join(data_dir, rollout_file), 'r', encoding='utf-8') as f:
        problem2groups = {}
        reader = jsonlines.Reader(f)
        for line in reader:
            # print(count)
            score_sum += line["score"]
            all_calling_times += line["call_counters"]
            max_calling_times = max(max_calling_times, line["call_counters"])
            max_length = max(max_length, len(line["sequences_str"]))
            questions.append(get_question(line["sequences_str"]))
            if get_question(line["sequences_str"]) not in problem2groups:
                problem2groups[get_question(line["sequences_str"])] = {}
                problem2groups[get_question(line["sequences_str"])]["score"] = []
                problem2groups[get_question(line["sequences_str"])]["multi_tool_calling_num"] = 0
                problem2groups[get_question(line["sequences_str"])]["calling_times"] = []
                problem2groups[get_question(line["sequences_str"])]["calling_costs"] = []
                problem2groups[get_question(line["sequences_str"])]["sequences"] = []
                problem2groups[get_question(line["sequences_str"])]["type"] = line
                problem2groups[get_question(line["sequences_str"])]["type"] = 'none' 
            if len(problem2groups[get_question(line["sequences_str"])]["score"]) == 8:
                print(f"duplicate question")
                continue
            if "em_score" in line["reason"]:
                problem2groups[get_question(line["sequences_str"])]["type"]="qa"
            elif "bad format" not in line["reason"]:
                problem2groups[get_question(line["sequences_str"])]["type"]="math"
            count += 1
            if line["score"] == -1:
                format_error_num += 1
            problem2groups[get_question(line["sequences_str"])]["score"].append(line["score"])
            problem2groups[get_question(line["sequences_str"])]["calling_times"].append(line["call_counters"])
            problem2groups[get_question(line["sequences_str"])]["calling_costs"].append(line["cost_counters"])
            problem2groups[get_question(line["sequences_str"])]["sequences"].append(line["sequences_str"])
            pattern = re.compile(r"<result>(.+?)</result>")
            match = pattern.findall(line["sequences_str"])
            if line["call_counters"]>=2:
                problem2groups[get_question(line["sequences_str"])]["multi_tool_calling_num"] +=1
                multi_tool_calling_num+=1
            if match:
                all_observation_length += sum([len(i) for i in match])
            if find_wrong(line["reason"],line["sequences_str"]):
                wrong_rollout_num+=1
                # wrong_rollout_sequences.append(line["sequences_str"].split("<|im_start|>assistant\n")[1])     
                wrong_rollout_sequences.append(line)        
            if len(problem2groups)==128 and count==1024:
                # print(f"break for {len(problem2groups)} and {count} in {rollout_step} rollout")
                break
            if len(problem2groups[get_question(line["sequences_str"])]["score"])> 8 :
                pass
    # print(f"finish for {len(problem2groups)} and {count} in {rollout_step} rollout")
    # 分析优势情况
    # assert len(problem2groups)==128,f"the problem2groups num is {len(problem2groups)} in {rollout_step} rollout"
    group_score_no_positive_num = 0 #reward没有正数
    group_score_no_positive_num_qa = 0 #reward没有正数，且问题类型为qa
    group_score_no_positive_num_math = 0 #reward没有正数，且问题类型为math
    group_advantage_zero_num = 0 #优势全是0
    group_advantage_zero_num_all_one = 0 #优势全是0，因为score全是1
    group_advantage_zero_num_all_zero = 0 #优势全是0，因为score全是0
    group_advantage_zero_num_all_zero_same_calling = 0 #group_advantage_zero_num_all_zero，且调用工具次数一致
    group_advantage_zero_num_all_negative_one = 0 #优势全是0，因为score全是-1
    group_advantage_zero_num_all_negative_one_same_calling = 0 #优势全是0，因为score全是-1，且调用工具次数一致
    group_score = dict([(key,problem2groups[key]["score"]) for key in problem2groups])
    group_oct_zero_num = 0
    group_oct_smooth_num = 0
    group_oct_zero_num_all_no_score = 0
    group_oct_zero_num_all_same_score = 0
    group_negative_oct_num = 0
    group_zero_calling_num_positive = 0#score>0且调用次数为0的数量
    group_multi_tool_calling_num = 0
    max_calling_times_num = 0
    qa_num = 0
    math_num = 0
    for question,score_list in group_score.items():
        assert len(score_list)==8,f"the score_list num is {len(score_list)} in {rollout_step} rollout for question {question}"
        calling_times_list = problem2groups[question]["calling_times"]
        calling_costs_list = problem2groups[question]["calling_costs"]
        multi_tool_calling_num = problem2groups[question]["multi_tool_calling_num"]
        max_calling_times_num += calling_times_list.count(max_calling_times)
        
        if multi_tool_calling_num > 0:
            group_multi_tool_calling_num +=1
        advantage_zero_num = 0 #当前问题的优势为0的轨迹数
        mean_score = sum(score_list)/len(score_list)
        oct_change_calling_list = [] #分数大于0的调用次数
        oct_change_calling_costs_list = [] #分数大于0的调用成本
        no_positive_calling_list = [] #分数小于等于0的调用次数
        oct_zero =False
        group_no_positive = True
        if question in questions_worng_dict:
            questions_worng_dict[question]["groups"].append(problem2groups[question])
        for idx,score in enumerate(score_list):
            if score-mean_score==0:
                advantage_zero_num+=1
            if score>0.3:
                oct_change_calling_list.append(calling_times_list[idx])
                oct_change_calling_costs_list.append(calling_costs_list[idx])
                group_no_positive = False
            else:
                no_positive_calling_list.append(calling_times_list[idx])
        group_score_no_positive_num += group_no_positive
        if problem2groups[question]["type"] == "qa":
            qa_num += 1
            if group_no_positive:
                group_score_no_positive_num_qa+=1
        elif problem2groups[question]["type"] == "math":
            math_num += 1
            if group_no_positive:
                group_score_no_positive_num_math+=1
        if 0 in oct_change_calling_costs_list:
            group_zero_calling_num_positive +=1
        if len(oct_change_calling_costs_list) == 0:
            group_oct_zero_num+=1
            group_oct_zero_num_all_no_score+=1
            oct_zero = True
        else:
            if len(set(oct_change_calling_costs_list))== 1:
                group_oct_zero_num+=1
                group_oct_zero_num_all_same_score+=1
                oct_zero = True
            else:
                if 0 in oct_change_calling_costs_list:
                    group_oct_smooth_num+=1
        if advantage_zero_num == len(score_list) and oct_zero: #可以排除全1但是调用工具有区别的
            group_advantage_zero_num+=1
            if score_list[0] == 1:
                group_advantage_zero_num_all_one += 1
            if score_list[0] == 0:
                group_advantage_zero_num_all_zero += 1
                if len(set(calling_times_list))==1:
                    group_advantage_zero_num_all_zero_same_calling+=1
            if score_list[0] == -1:
                group_advantage_zero_num_all_negative_one += 1
                if len(set(calling_times_list))==1:
                    group_advantage_zero_num_all_negative_one_same_calling+=1
        if advantage_zero_num == len(score_list) and not oct_zero:
            pass
        if advantage_zero_num!=len(score_list) and len(set(no_positive_calling_list))>0 and len(set(oct_change_calling_list))==1:#说明得分有差距
            #如果加上negative的oct会增加的oct触发数量
            #说明在一组里面有最优的调用轨迹，且有得分小于等于0的轨迹,且这些轨迹存在调用次数差
            optim_call = min(oct_change_calling_list)
            # print(f"score_list: {score_list}")
            # print(f"calling_times_list: {calling_times_list}")
            # print(f"optim_call: {optim_call}")
            # print(f"no_positive_calling_list: {no_positive_calling_list}")
            if len(set(no_positive_calling_list))==1 and no_positive_calling_list[0] == optim_call:
                pass
            else:
                group_negative_oct_num+=1
            pass
    
    rollout_step2metrics[int(rollout_step)] = {
        "score": score_sum / count,  # 平均分数
        "max_calling_times": max_calling_times,  #这次rollout的最大调用次数
        "max_calling_times_num": max_calling_times_num,
        "all_calling_times": all_calling_times, #这次rollout的总调用次数
        "group_zero_calling_num_positive": group_zero_calling_num_positive, #这次rollout里面有多少个问题存在调用次数为0但是分数为大于0的分数
        "max_length": max_length, 
        "all_observation_length": all_observation_length,
        # "question": questions,
        "wrong_rollout_num": wrong_rollout_num,
        "format_error_num": format_error_num,
        "wrong_rollout_sequences": wrong_rollout_sequences,
        "multi_tool_calling_num": multi_tool_calling_num,
        "group_score_no_positive_num": group_score_no_positive_num,
        "group_score_no_positive_num_math":group_score_no_positive_num_math,
        "group_score_no_positive_num_qa":group_score_no_positive_num_qa,
        "group_oct_zero_num": group_oct_zero_num,
        "group_oct_smooth_num":group_oct_smooth_num,
        "group_negative_oct_num":group_negative_oct_num,
        "group_oct_zero_num_all_same_score": group_oct_zero_num_all_same_score,
        "group_oct_zero_num_all_no_score": group_oct_zero_num_all_no_score,
        "group_advantage_zero_num": group_advantage_zero_num,
        "group_advantage_zero_num_all_one": group_advantage_zero_num_all_one,
        "group_advantage_zero_num_all_zero": group_advantage_zero_num_all_zero,
        "group_advantage_zero_num_all_negative_one":group_advantage_zero_num_all_negative_one,
        "group_multi_tool_calling_num":group_multi_tool_calling_num,
        "group_advantage_zero_num_all_zero_same_calling":group_advantage_zero_num_all_zero_same_calling,
        "group_advantage_zero_num_all_negative_one_same_calling":group_advantage_zero_num_all_negative_one_same_calling,
        "qa_num": qa_num,
        "math_num":math_num        
        }
print(expertment_name)
result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"analyse",expertment_name)
os.makedirs(result_dir, exist_ok=True)
x = [int(i) for i in rollout_step2metrics.keys()]
x = sorted(x)

y = [rollout_step2metrics[i]["format_error_num"] for i in x]
draw_with_max(x,y,result_dir,"format_error_num")


y = [rollout_step2metrics[i]["wrong_rollout_num"] for i in x]
draw_with_max(x,y,result_dir,"wrong_rollout_num")

y = [rollout_step2metrics[i]["max_length"] for i in x]
draw_with_max(x,y,result_dir,"max_length")

y = [rollout_step2metrics[i]["max_calling_times"] for i in x]
draw_with_max(x,y,result_dir,"max_calling_times")

y = [rollout_step2metrics[i]["max_calling_times_num"] for i in x]
draw_with_max(x,y,result_dir,"max_calling_times_num")

y = [rollout_step2metrics[i]["all_calling_times"] for i in x]
draw_with_max(x,y,result_dir,"all_calling_times")

y = [rollout_step2metrics[i]["all_observation_length"] for i in x]
draw_with_max(x,y,result_dir,"all_observation_length")

y = [rollout_step2metrics[i]["group_multi_tool_calling_num"] for i in x]
draw_with_max(x,y,result_dir,"group_multi_tool_calling_num")

y = [rollout_step2metrics[i]["group_zero_calling_num_positive"] for i in x]
draw_with_max(x,y,result_dir,"group_zero_calling_num_positive")

y = [rollout_step2metrics[i]["group_score_no_positive_num"] for i in x]
draw_with_max(x,y,result_dir,"group_score_no_positive_num")

y = [rollout_step2metrics[i]["group_negative_oct_num"] for i in x]
draw_with_max(x,y,result_dir,"group_negative_oct_num")
y1 = [128-rollout_step2metrics[i]["group_oct_zero_num"] for i in x]
y2 = [128-rollout_step2metrics[i]["group_oct_zero_num"]+rollout_step2metrics[i]["group_negative_oct_num"] for i in x]
draw_multi_lines(x,[y,y1,y2],
                 ["group_negative_oct_num","group_positive_oct_num","group_oct_num"],result_dir,"group_advantage_oct_num_multi")

y = [rollout_step2metrics[i]["math_num"] for i in x]
y1 = [rollout_step2metrics[i]["qa_num"] for i in x]
y2 = [rollout_step2metrics[i]["group_score_no_positive_num_math"] for i in x]
y3 = [rollout_step2metrics[i]["group_score_no_positive_num_qa"] for i in x]
draw_multi_lines(x,[y,y1,y2,y3],
                 ["math_num","qa_num","group_score_no_positive_num_math","group_score_no_positive_num_qa"],result_dir,"math_num_num_multi")



y = [rollout_step2metrics[i]["group_advantage_zero_num"] for i in x]
y1 = [rollout_step2metrics[i]["group_advantage_zero_num_all_one"] for i in x]
y2 = [rollout_step2metrics[i]["group_advantage_zero_num_all_zero"] for i in x]
y3 = [rollout_step2metrics[i]["group_advantage_zero_num_all_negative_one"] for i in x]
y4 = [rollout_step2metrics[i]["group_score_no_positive_num"] for i in x]
draw_with_max(x,y,result_dir,"group_advantage_zero_num")
draw_multi_lines(x,[y,y1,y2,y3,y4],
                 ["group_advantage_zero_num","group_advantage_zero_num_all_one","group_advantage_zero_num_all_zero",
                  "group_advantage_zero_num_all_negative_one","group_score_no_positive_num"],result_dir,"group_advantage_zero_num_multi")


y = [rollout_step2metrics[i]["group_advantage_zero_num_all_negative_one"] for i in x]
y1 = [rollout_step2metrics[i]["group_advantage_zero_num_all_zero"] for i in x]
y2 = [rollout_step2metrics[i]["group_advantage_zero_num_all_zero_same_calling"] for i in x]
y3 = [rollout_step2metrics[i]["group_advantage_zero_num_all_negative_one_same_calling"] for i in x] 
draw_multi_lines(x,[y,y1,y2,y3],
                 ["group_advantage_zero_num_all_negative_one","group_advantage_zero_num_all_zero","group_advantage_zero_num_all_zero_same_calling","group_advantage_zero_num_all_negative_one_same_calling"],result_dir,"group_advantage_zero_calling_num_multi")


y = [rollout_step2metrics[i]["group_oct_zero_num"] for i in x]
draw_with_max(x,y,result_dir,"group_oct_zero_num")
y = [128-rollout_step2metrics[i]["group_oct_zero_num"] for i in x]
y1 = [128-rollout_step2metrics[i]["group_advantage_zero_num"] for i in x]
y2 = [rollout_step2metrics[i]["group_oct_smooth_num"] for i in x]
draw_multi_lines(x,[y,y1,y2],
                 ["group_oct_no_zero_num","group_advantage_no_zero_num","group_oct_smooth_num"],result_dir,"group_advantage_no_zero_num_multi")


y = [ rollout_step2metrics[i]["group_oct_smooth_num"] for i in x]
draw_with_max(x,y,result_dir,"group_oct_smooth_num")
y = [rollout_step2metrics[i]["group_oct_smooth_num"]/(128-rollout_step2metrics[i]["group_oct_zero_num"]+0.00001) for i in x]
draw_with_max(x,y,result_dir,"group_oct_smooth_num_radio")
# draw_multi_lines(x,[y,y1],
#                  ["group_oct_smooth_num","group_oct_smooth_num_radio"],result_dir,"group_advantage_oct_smooth_num_multi")

# print(sorted(rollout_step2metrics.items(), key=lambda x: x[0]))
with open(os.path.join(result_dir,f"rollout_step2metrics.json"), "w") as f:
    # json.dump(sorted(rollout_step2metrics.items(), key=lambda x: x[0]), f,indent=4)
    json.dump(rollout_step2metrics, f,indent=4)
if specific_rollout_iter_num != -1:
    with open(os.path.join(result_dir,f"rollout_step2metrics_specific_iter_{specific_rollout_iter_num}.json"), "w") as f:
        json.dump(rollout_step2metrics[specific_rollout_iter_num], f,indent=4) 
with open(os.path.join(result_dir,f"llm_wrong_question_result.json"), "w") as f:
    new_questions_worng_dict={}
    for question in questions_worng_dict:
        groups = questions_worng_dict[question]["groups"]
        score_sum = 0
        for group in groups:
            score_sum +=sum(group["score"]) 
        if question not in new_questions_worng_dict:
            new_questions_worng_dict[question] = {}
        new_questions_worng_dict[question]["score_sum"] = score_sum         
    json.dump(questions_worng_dict, f,indent=4)

#### 分析search
# search_questions = get_search_questions(data_dir,result_dir)
# print(len(search_questions))
# with open(os.path.join(result_dir,"search_questions.json"), "w",encoding="utf-8") as f:
#     json.dump(list(search_questions), f, ensure_ascii=False, indent=4)
#     print(f'save to {os.path.join(result_dir,"search_questions.json")}')
# all_search_questions = set()
# rollout_dir = "/share/home/jfliang/Project/sxjiang/Tool-Star-OCT/Tool_Star_RL/verl_checkpoints"
# result_dir = os.path.dirname(os.path.abspath(__file__))
# all_search_questions = get_search_questions(rollout_dir,result_dir,except_experiment=expertment_name,resume=True)
# print("old search questions num:",len(all_search_questions))
# print("resume search questions num:",len(all_search_questions.intersection(search_questions)))