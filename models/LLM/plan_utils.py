import random
import re
def max_sum_under_limit(nums, N):
    dp = [0] * (N)
    choice = [-1] * (N) 

    for idx, num in enumerate(nums):
        for j in range(N - 1, num - 1, -1):  
            if dp[j - num] + num > dp[j]:  
                dp[j] = dp[j - num] + num
                choice[j] = idx 

    max_sum = 0
    for i in range(N - 1, -1, -1): 
        if dp[i] > max_sum:
            max_sum = dp[i]
            break

    selected_indices = []
    j = max_sum
    while j > 0 and choice[j] != -1:
        selected_indices.append(choice[j])
        j -= nums[choice[j]] 

    selected_indices.reverse() 

    return selected_indices, max_sum

def minimize_time(subtasks):
    num_a_subtasks = 0
    num_subtasks = 0

    a_id_list = []
    a_time_list = []

    b_time = 0
    b_id = None

    for subtask in subtasks:
        num_subtasks += 1
        if subtask["type"] == "B":
            b_time = subtask["time"]
            b_id = subtask["subtask_id"]
        elif subtask["type"] == "A":
            num_a_subtasks += 1
            a_id_list.append(subtask["subtask_id"])
            a_time_list.append(subtask["time"])

    if num_a_subtasks == num_subtasks:
        return a_id_list, sum(a_time_list)
    
    selected_indices, max_sum = max_sum_under_limit(a_time_list, b_time)
    all_indices = list(range(len(a_time_list)))
    left_indices = list(set(all_indices) - set(selected_indices))

    if left_indices:
        additional_a_time = sum(a_time_list[i] for i in left_indices)
        b_time += additional_a_time  

    selected_a_ids = [a_id_list[i] for i in selected_indices]
    left_a_ids = [a_id_list[i] for i in left_indices]

    if not left_a_ids:
        min_time_action_list = [b_id] + selected_a_ids + [b_id]
    else:
        b_block = [b_id] + selected_a_ids + [b_id]

        insert_position = 0
        for i, task_id in enumerate(left_a_ids):
            if task_id > b_id:
                insert_position = i
                break
        else:
            insert_position = len(left_a_ids)

        for i, item in enumerate(b_block):
            left_a_ids.insert(insert_position + i, item)

        min_time_action_list = left_a_ids

    return min_time_action_list, b_time


def build_subtask_list(cont_ids, non_cont_ids, gt_subtasks):
    filtered_list = []
    filtered_cont_ids = []
    filtered_non_cont_ids = []

    for i in cont_ids:
        idx = i - 1
        if 0 <= idx < len(gt_subtasks):
            filtered_list.append({
                "subtask_id": i,
                "type": "A",
                "time": gt_subtasks[idx]["time"]
            })
            filtered_cont_ids.append(i)

    for i in non_cont_ids:
        idx = i - 1
        if 0 <= idx < len(gt_subtasks):
            filtered_list.append({
                "subtask_id": i,
                "type": "B",
                "time": gt_subtasks[idx]["time"]
            })
            filtered_non_cont_ids.append(i)

    return filtered_list, filtered_cont_ids, filtered_non_cont_ids


def extract_subtasks(text):
    pattern = r"Subtasks that require continuous attention to operate:\s*(.*?)\.\nSubtasks that do not need continuous attention to operate:\s*(.*?)\."
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return [], []
    
    continuous_part, non_continuous_part = match.groups()


    continuous_ids = [int(x) for x in re.findall(r"subtask\s*(\d+)", continuous_part)]
    non_continuous_ids = [int(x) for x in re.findall(r"subtask\s*(\d+)", non_continuous_part)]


    return continuous_ids, non_continuous_ids

def get_text_plan(type_a_list, type_b_list, min_time_action_list):
    if type_b_list != []:
        b_id = [num for num in set(min_time_action_list) if min_time_action_list.count(num) > 1][0]
        task_sequence = []
        b_start = False
        for task_id in min_time_action_list:
            if task_id == b_id:
                if b_start == False:
                    b_start = True
                    task_sequence.append(f"start subtask {task_id}")
                else:
                    task_sequence.append(f"recheck subtask {task_id}")
            else:

                task_sequence.append(f"subtask {task_id}")

        task_sequence_str = ", ".join(task_sequence)

        text_plan = "I will complete the subtasks in the optimal order: " + task_sequence_str + ".\n"

    else:
        text_plan = "I will complete the subtasks in the optimal order: " + ", ".join([f"subtask {i}" for i in min_time_action_list]) + ".\n"

    return text_plan

def get_gt_subtasks_from_question(question):
    pattern = r"\(\d+\)\s(.*?)(\(\d+\s*minutes\))"
    matches = re.findall(pattern, question)

    subtasks = []
    for match in matches:
        task_text = match[0].strip()
        time_match = re.search(r"\((\d+)\s*minutes\)", match[1])
        time = int(time_match.group(1)) if time_match else None
        subtasks.append({
            "text": task_text,
            "time": time
        })
    
    return subtasks

def call_task_planner(reasoning, question):
    cont_ids, non_cont_ids = extract_subtasks(reasoning)
    gt_subtasks = get_gt_subtasks_from_question(question) 
    subtask_list, cont_ids, non_cont_ids = build_subtask_list(cont_ids, non_cont_ids, gt_subtasks) 
    min_time_action_list, _ = minimize_time(subtask_list)
    text_plan = get_text_plan(cont_ids, non_cont_ids, min_time_action_list)
    return text_plan

