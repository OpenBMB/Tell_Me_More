import json
from cprint import cprint
import copy


file_path = "../data/IN3/test.jsonl"
save_path = "../data/data_labeling/test_data_report.jsonl"

with open(file_path, "r") as file:
    data = [json.loads(line) for line in file.readlines()]

quit_flag = False

# Please specify the range of data to label
for d in data[:]:
    ori_d = copy.deepcopy(d)
    begin_current = True
    print("-=-=-=-=-=-=- Begin a new task -=-=-=-=-=-=-")
    while begin_current:
        cprint.err(f"Category: {d['category']}\nTask: {d['task']}")
        vague = "y" if d["vague"] else "n"
        init = "vague (y)" if d["vague"] else "clear (n)"
        cprint.ok(f"GPT believe the task is {init}.")
        if vague == "y":
            middle = "GPT suggests some missing details are:\n"
            for detail in d["missing_details"]:
                middle += f'{detail["description"]}, {detail["importance"]}\n'
            cprint.info(middle)
        
        user_vague = input("Do you think the task is vague (y) or clear (n)?\nYour Response: ").strip()
        d["user_vague"] = True if user_vague == "y" else False
        
        d["user_reject"] = []
        d["user_approve"] = []
        d["user_rectify"] = []
        d["user_add"] = []
        
        user_missing_details = copy.deepcopy(d["missing_details"])
        for detail in user_missing_details:
            detail.pop("inquiry", None)
            detail.pop("options", None)
        
        if user_vague == "n" and vague == "y":
            for detail in user_missing_details:
                d["user_reject"].append(detail)
        if user_vague == "y":
            if vague == "y":
                cprint.ok("I will present the GPT's suggestions one by one.\n1. Directly press Enter if you totally agree with suggestion\n2. Press 'x' if you think the this missing detail is unnecessary and reject it.\n3. If you think you should make rectification to description or the score, copy paste it and make retification. Please use ',' to separate your rectified description and score.\nNow please decide one by one:")
                for detail in user_missing_details:
                    desc = detail["description"]
                    importance = detail["importance"]
                    cprint.info(f"{desc}, {importance}")
                    user_input = input("Your Response: ").strip()
                    if user_input == "":
                        d["user_approve"].append(detail)
                    elif user_input == "x":
                        d["user_reject"].append(detail)
                    else:
                        desc, importance = user_input.split(",")
                        d["user_rectify"].append({"description": desc.strip(), "importance": importance.strip()})
            cprint.ok("Please add any missing details you think are necessary. Please use ',' to separate your description and score, one in a line.\nPress 'q' to quit.")
            while True:
                user_input = input("Additional Details: ").strip()
                if user_input == "q":
                    break
                desc, importance = user_input.split(",")
                d["user_add"].append({"description": desc.strip(), "importance": importance.strip()})
                cprint.info("Successfully Parsed and Added")
                
        cprint.ok("Please confirm all your decisions in this round. Press Enter to continue, press 'q' to quit, or press 'r' to restart this round.")
        user_input = input("Your Response: ").strip()
        if user_input == "r":
            d = copy.deepcopy(ori_d)
            continue
        
        # save this round
        cprint.warn("Saving this round ...")
        with open(save_path, "a") as file:
            file.write(json.dumps(d) + "\n")
        begin_current = False
        if user_input == "q":
            quit_flag = True
    
    if quit_flag:
        break
        
cprint.fatal("All Done!")

    