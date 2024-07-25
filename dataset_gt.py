import json
import os
import random

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT
from econml.dml import LinearDML

from util import has_confounder,Relationship,has_collider
to_write = []

name = "ate"
# os.remove('./dataset_gt.json')
with open(f"./dataset_{name}.json","r",encoding='utf-8') as f:
    for line in f:
        to_write.append(json.loads(line))

# print(to_write)
question_CG = {
    "IT":
        {
            "whether {} and {} is independent.",
            "Is {} independent of {}?",
            "Are {} and {} statistically independent?",
            "Does the occurrence of {} independent on {}, or vice versa?",
            "Can we assert {} and {} are independent, or are they related?",
            "Can we consider {} and {} as independent events?",
            "Do {} and {} independent and don't have any influence on each other?",
            "Is there no statistically correlation between {} and {}?",
            "test whether Are {} and {} statistically unrelated or dependent?",
            "Test the independence of {} and {}."
        },
    "CIT":
        {
            "whether {} and {} is independent under condition {}?",
            "Is {} independent of {} given condition {}?",
            "Are {} and {} statistically independent given the condition {}?",
            "Does the independence of {} and {} hold true under condition {}?",
            "Can we consider {} and {} as conditionally independent with respect to {}?",
            "Is the independence between {} and {} maintained given the condition {}?",
            "Are {} and {} conditionally independent with the presence of condition {}?",
            "Can we assume that {} and {} are independent given the condition {}?",
            "Is the independence of {} and {} upheld in the presence of condition {}?",
            "Does the independence between {} and {} persist under the condition {}?"
         },
    "MULTCIT" :
        {
            "whether {} and {} is independent under conditions : ",
            "Determine the independence of {} and {} given the following conditions : ",
            "Examine if {} and {} are independent under the specified conditions : ",
            "Assess the independence between {} and {} with the provided conditions : ",
            "Investigate whether {} and {} exhibit independence given the outlined conditions : ",
            "Explore the independence of {} and {} under the given circumstances : ",
            "Ascertain if there is independence between {} and {} given the stated conditions : ",
            "Check for independence between {} and {} based on the conditions described : ",
            "Verify the independence status of {} and {} under the listed conditions : ",
            "Evaluate the independence of {} and {} under the mentioned conditions : ",
            "Examine whether {} and {} are independent, considering the provided conditions : "
        },
    "CAUSE" :
        {
            "whether {} directly cause {}.",
            "Assess if {} has a direct causal impact on {}.",
            "Examine the direct causation relationship.if {} directly cause {}?",
            "Investigate whether {} directly influences {}.",
            "Evaluate if there exists the direct causal connection from {} to {}.",
            "Scrutinize if {} leads to a direct causation of {}.",
            "Determine whether {} is a direct cause of {}.",
            "Assess if there is the direct causal link of {} to {}.",
            "Verify if {} directly results in the causation of {}."
        },
    "Has-Collider" :
        {
            "Whether there exists at least one collider (i.e., common effect) of {} and {}",
            "Determine if there is at least one common effect (collider) of both {} and {}.",
            "Assess the presence of a shared outcome, serving as a collider, for variables {} and {}.",
            "Examine the potential existence of a shared consequence as a collider for {} and {}.",
            "Evaluate if {} and {} share a common effect (collider).",
            "Analyze the presence of a common outcome serving as a collider for {} and {}.",
            "Verify if there exists a shared effect, acting as a collider, for both {} and {}.",
            "Explore whether a common consequence is a collider for variables {} and {}.",
            "Assess the existence of at least one common effect (collider) between {} and {}."
        },
    "Has-Confounder" :
        {
            "There exists at least one confounder (i.e., common cause) of {} and {}.",
            "Confirm the presence of at least one common cause (confounder) influencing both {} and {}.",
            "Verify whether there exists a shared factor, acting as a confounder, for variables {} and {}.",
            "Examine the potential existence of a common cause (confounder) impacting both {} and {}.",
            "Assess if {} and {} share at least one confounding factor (common cause).",
            "Scrutinize the presence of a shared influencing factor, serving as a confounder, for {} and {}.",
            "Investigate whether there is at least one confounder affecting both {} and {}.",
            "Analyze the potential impact of a common cause (confounder) on variables {} and {}.",
            "Verify the presence of a shared influencing factor, acting as a confounder, for {} and {}.",
            "Explore whether a common factor is a confounder for variables {} and {}.",
            "Evaluate the existence of at least one confounder (common cause) between {} and {}."
    },
    "CAUSALKG" :
        {
            "please generate causal graph of the input tabular data.",
            "Produce a causal graph representing the relationships within the given tabular data.",
            "Generate a directed graph that illustrates the causal connections inherent in the provided tabular dataset.",
            "Create a graphical model depicting the causality among variables in the input tabular data.",
            "Construct a causal diagram illustrating the interdependencies among the variables in the tabular dataset.",
            "Formulate a graph that visually represents the cause-and-effect relationships present in the input tabular information.",
            "Develop a graphical representation outlining the causal structure of the tabular data.",
            "Build a directed acyclic graph (DAG) that reflects the causal influences within the input tabular dataset.",
            "Establish a graphical model showcasing the causal links between variables derived from the tabular data.",
            "Design a causal graph that visually captures the cause-and-effect relationships inherent in the tabular information.",
            "Construct a directed graph that visually displays the causal pathways within the given tabular dataset."
        },
    "PARTIAL_CG":
        {
            "Please generate a partial causal diagram for some of the following variables that interest me : ",
            "Generate a subset of a causal diagram for the variables of interest : ",
            "Create a partial graphical model illustrating causal relationships among selected variables : ",
            "Develop a restricted causal graph focusing on specific variables from the given set : ",
            "Formulate a partial directed acyclic graph (DAG) depicting causal connections for chosen variables : ",
            "Construct a limited causal diagram featuring only the variables of interest : ",
            "Produce a subsection of a graphical model, emphasizing the causal links within the selected variables : ",
            "Build a causal graph subset, emphasizing relationships among the variables you find intriguing : ",
            "Develop a focused causal diagram, highlighting causal connections for the specified variables : ",
            "Form a segment of a directed graph that visually represents causal relationships among chosen variables : ",
            "Create a restricted causal network, showcasing the partial causal influences among the variables of interest : "
        },
    "ATE":{

    }
}

csv_files = os.listdir("./data")

final_list = []

import pandas as pd
import numpy as np

skip_num = 0
def get_name(to_find,name_list):
    return [name_list.index(ind) for ind in to_find]

for id,line in enumerate(to_write):
    if id < skip_num:
        continue
    Q = line["question_type"]
    nd_num = line["node num"]
    csv_name = random.choice([i for i in csv_files if i.startswith("{}_".format(nd_num))])
    line['file'] = csv_name
    sampled_elements = line['variables']
    interest = line["interest"]
    data_df = pd.read_csv(f'./data/{csv_name}', header=None)
    data_df.columns = sampled_elements
    data = np.array(data_df)
    if Q in ["IT", "CAUSE", "Has-Collider", "Has-Confounder"]:
        Q_content = random.choice(list(question_CG[Q]))
        question = Q_content.format(interest[0], interest[1])
        line["Q"] = question
        if Q == "IT":
            cit = CIT(data=data, method='fisherz')
            pValue = cit(data_df.columns.get_loc(interest[0]), data_df.columns.get_loc(interest[1]))
            if pValue < 0.05:
                line['gt'] = 'yes'
            else:
                line['gt'] = 'no'
            with open(f"./dataset_{name}_gt.json", 'a+') as f:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')
        if Q == "CAUSE":
            cg = pc(data, 0.05)
            i = data_df.columns.get_loc(interest[0])
            j = data_df.columns.get_loc(interest[1])
            if cg.G.graph[i][j] == -1 and cg.G.graph[j][i] == 1:
                line["gt"] = "yes"
            elif cg.G.graph[j][i] == -1 and cg.G.graph[i][j] == -1:
                line['gt'] = 'uncertain'
            else:
                line['gt'] = 'no'
            with open(f"./dataset_{name}_gt.json", 'a+') as f:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')
        if Q == "Has-Confounder":
            cg = pc(data, 0.05)
            i = data_df.columns.get_loc(interest[0])
            j = data_df.columns.get_loc(interest[1])
            rela,tmp = has_confounder(i, j, cg.G.graph)
            if rela == Relationship.YES:
                line["gt"] = "yes"
            elif rela == Relationship.UNCERTAIN:
                line['gt'] = 'uncertain'
            else:
                line['gt'] = 'no'
            with open(f"./dataset_{name}_gt.json", 'a+') as f:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')
        if Q == "Has-Collider":
            cg = pc(data, 0.05)
            i = data_df.columns.get_loc(interest[0])
            j = data_df.columns.get_loc(interest[1])
            rela, tmp = has_collider(i, j, cg.G.graph)
            if rela == Relationship.YES:
                line["gt"] = "yes"
            elif rela == Relationship.UNCERTAIN:
                line['gt'] = 'uncertain'
            else:
                line['gt'] = 'no'
            with open(f"./dataset_{name}_gt.json", 'a+') as f:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')
    if Q == "CIT" or Q == "MULTCIT":
        Q_content = random.choice(list(question_CG[Q]))
        if Q == "CIT":
            question = Q_content.format(interest[0], interest[1],interest[2])
            line["Q"] = question
        elif Q == "MULTCIT":
            question = Q_content.format(interest[0], interest[1]) + ','.join(interest[2:])
            line["Q"] = question

        cit = CIT(data=data, method='fisherz')
        pValue = cit(data_df.columns.get_loc(interest[0]), data_df.columns.get_loc(interest[1]),[data_df.columns.get_loc(col) for col in interest[2:]])
        if pValue < 0.05:
            line['gt'] = 'yes'
        else:
            line['gt'] = 'no'
        with open(f"./dataset_{name}_gt.json", 'a+') as f:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
    if Q == "CAUSALKG":
        Q_content = random.choice(list(question_CG[Q]))
        line["Q"] = Q_content
        cg = pc(data, 0.05,node_names=line["variables"])
        line['gt'] = f'{name}_{id}.txt'
        with open(f"./dataset_{name}_gt.json", 'a+') as f:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
        with open(os.path.join('causal_graph',f'{name}_{id}.txt'),"w") as f:
            f.write(str(cg.G))
    if Q == "PARTIAL_CG":
        Q_content = random.choice(list(question_CG[Q]))
        Q_content = Q_content+','.join(interest)
        line["Q"] = Q_content
        cg = pc(np.array(data_df[line['interest']]), 0.05,node_names=line['interest'])
        line['gt'] = f'{name}_{id}.txt'
        with open(f"./dataset_{name}_gt.json", 'a+') as f:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
        with open(os.path.join('causal_graph',f'{name}_{id}.txt'),"w") as f:
            f.write(str(cg.G))
    if Q == "ATE":
        est = LinearDML(random_state=123)
        inds = get_name([i for i in line['variables'] if i not in line['interest']],line['variables'])
        Q_content = "calculate the Average Treatment Effect (ATE) of a continuous treatment variable  {T} on an outcome variable {Y}, given that the treatment {T} change from {T0} to {T1}."
        line["Q"] = Q_content.format(T=interest[0], Y=interest[1],T0=line['T0'],T1=line['T0']) + "The remaining variables can be regarded as confounder."
        est.fit(data[:,line['variables'].index(interest[1])],data[:,line['variables'].index(interest[0])],X=np.array(data_df.iloc[:,inds]))
        line['gt'] = est.ate(T0=line['T0'],T1=line["T1"],X=np.array(data_df.iloc[:,inds]))
        with open(f"./dataset_{name}_gt.json", 'a+') as f:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')