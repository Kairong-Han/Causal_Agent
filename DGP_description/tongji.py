import json

jsons = []

with open("./dataset_gt.json","r") as f:
    for lines in f:
        jsons.append(lines)

dir = {
    "IT":[0,0,0],
    "CIT":[0,0,0],
    "MULTCIT":[0,0,0],
    "CAUSE":[0,0,0],
    "Has-Collider":[0,0,0],
    "Has-Confounder":[0,0,0]
}

ans = {
    "yes":0,
    "no":1,
    "uncertain":2
}

for line in jsons:
    line = json.loads(line)
    if line["question_type"] not in ['CAUSALKG',"PARTIAL_CG"]:
        dir[line["question_type"]][ans[line["gt"]]] += 1

print(dir)