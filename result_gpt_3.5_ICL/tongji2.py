import json
for target in  ["IT","CIT","MULTCIT","CAUSE","Has-Collider","Has-Confounder","CAUSALKG","PARTIAL_CG"]:
    print(f"======{target}=======")
    res = []
    name = [str(i) for i in range(3,11)]
    # name.extend(['-1'])

    with open("market_icl_result/jiedianshu.jsonl", 'r') as f:
        for line in f:
            a_line = json.loads(line)
            if a_line['name'] in [f"{target}.jsonl",f'marketing_elements_{target}.jsonl']:
                res.append(a_line)
    with open("medical_icl_result/jiedianshu.jsonl", 'r') as f:
        for line in f:
            a_line = json.loads(line)
            if a_line['name'] in [f"{target}.jsonl",f'marketing_elements_{target}.jsonl']:
                res.append(a_line)
    correct = 0
    total = 0
    for i in name:
        if i not in res[0]:
            continue
        mylist = [k[i] for k in res]
        print(mylist)
        correct += mylist[0][0]+mylist[1][0]
        total += mylist[0][1]+mylist[1][1]
        acc = (mylist[0][0]+mylist[1][0])/(mylist[0][1]+mylist[1][1])
        print(i,acc)
    print(correct,total,correct/total)
