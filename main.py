import io
import json
import time

import pandas as pd
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.ANM.ANM import ANM
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.utils.cit import CIT
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
# from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.utils.GraphUtils import GraphUtils
from econml.dr import LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.orf import DMLOrthoForest
from econml.sklearn_extensions.linear_model import WeightedLassoCV
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_community.chat_models import ChatOpenAI
import os
from matplotlib import image as mpimg, pyplot as plt

import warnings
from langchain.agents import Tool
import matplotlib
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import (Lasso, LassoCV, LogisticRegression,
                                  LogisticRegressionCV, LinearRegression,
                                  MultiTaskElasticNet, MultiTaskElasticNetCV)
import sys

from sklearn.preprocessing import PolynomialFeatures
from DGP_description.util import has_confounder, has_collider, Relationship, Dataloader, CGMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')

# 定义工具


folder = 'DGP_description/data'
ate_source_folder = 'ATE_source/data'
name_out_to_in = []
CG_out_dir = './temp_CG'
skip = ''

api_key = ''
outer_item = []
cit_method_name = ''

from langchain_community.llms.chatglm3 import ChatGLM3



data = None
col_name = None
memory = None


def condition_independent_test(input_str):
    try:
        global data, col_name
        print(input_str)
        input_str = json.loads(input_str)
        index = input_str['interesting_var']

        condition = None if input_str['condition'] == "None" or [] else input_str['condition']
        postfix = ''
        if condition is not None:
            postfix = 'under conditions : ' + ','.join(condition)
        cit = CIT(data=np.array(data), method='fisherz')
        pValue = cit(data.columns.get_loc(index[0]), data.columns.get_loc(index[1]),
                     [data.columns.get_loc(col) for col in condition] if condition is not None else None)

        if pValue < 0.05:
            return '''{} and {} is independent'''.format(index[0], index[1])+postfix
        else:
            return '''{} and {} is  not independent'''.format(index[0], index[1])+postfix
    except Exception as e:
        return f"tool raise a error :{e}. please check your input format and input variables."

def draw_pydot(pyd):
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def generate_causalgraph(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str)
        filename = input_str['filename']
        method = input_str['method'] if 'method' in input_str.keys() else 'pc'
        interested_var = input_str['interesting_var'] if 'interesting_var' in input_str.keys() else None
        analyse_relationship = input_str['analyse_relationship'] if 'analyse_relationship' in input_str.keys() else None
        config = input_str['config'] if 'config' in input_str.keys() else None
        print(config)
        data_sum = pd.read_csv(os.path.join(folder, filename), header=None)
        data_sum.columns = name_out_to_in
        data = np.array(data_sum)
        if interested_var is not None and interested_var != [] and analyse_relationship == "False":
            data = np.array(data_sum[interested_var])
            cg = pc(data, 0.05, fisherz, node_names=interested_var)
        else:
            cg = pc(data, 0.05, fisherz, node_names=name_out_to_in)
        # cg.draw_pydot_graph()
        name = memory.Get_name(filename)
        memory.add(name, cg)
        with open("tempCG.txt", "w") as f:
            f.write(str(cg.G))
        return f"causal graph named '{name}' is generate succeed! and have written to the memory, you can use {name} as parameter 'cg_name' "
    except Exception as e:
        return str(e)


def prase_data( config, filename):
    global folder,name_out_to_in
    Y_name = config['Y'] if 'Y' in config.keys() and config['Y'] != [] else None
    Z_name = config['Z'] if 'Z' in config.keys() and config['Z'] != [] else None
    T_name = config['T'] if 'T' in config.keys() and config['T'] != [] else None
    W_name = config['W'] if 'W' in config.keys() and config['W'] != [] else None
    X_name = config['X'] if 'X' in config.keys() and config['X'] != [] else None
    assert Y_name is not None and T_name is not None
    data = pd.read_csv(os.path.join(folder, filename), header=None)
    data.columns = name_out_to_in
    T_mat = data[T_name] if T_name is not None else None
    Y_mat = data[Y_name] if Y_name is not None else None
    Z_mat = data[Z_name] if Z_name is not None else None
    W_mat = data[W_name] if W_name is not None else None
    X_mat = data[X_name] if X_name is not None else None
    return T_mat, Y_mat, Z_mat, W_mat, X_mat,config['T0'],config['T1']


def ATE_sim(input_str):
    try:
        print("\n" + input_str)
        input_str = json.loads(input_str)
        filename = input_str['filename']
        config = input_str['config']

        # T0 = input_str['T0']
        # T1 = input_str['T1']
        # print(T0,T1)
        T_mat, Y_mat, Z_mat, W_mat, X_mat,T0,T1 = prase_data(config, filename)
        est = LinearDML(random_state=123)
        est.fit(Y_mat,T_mat,X=X_mat,W=W_mat)
        ate = est.ate(T0=T0,T1=T1,X=X_mat)
        return f'ate : E(Y|T1) - E(Y|T0) is {ate}'
    except Exception as e:
        return f"tools error : {str(e)}"


def empty(input_str):
    print(input_str)
    return "this question doesn't need other tools, so you can only answer the question directly."


from causallearn.search.FCMBased.ANM.ANM import ANM


def Determine_collider(input_str):
    global folder, name_out_to_in, memory
    try:
        print("\n" + input_str)
        input_str = json.loads(input_str)
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        filename = None
        cg = None
        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        rela, varlist = has_collider(index_a, index_b, cg.G.graph)

        if rela == Relationship.YES:
            return "There exists at least one collider {} of {} and {} ".format(node_names[varlist[0]],
                                                                                activate_var[0],
                                                                                activate_var[1])
        if rela == Relationship.UNCERTAIN:
            return "Whether there exist collider of {} and {} is uncertain because some edge direction is uncertain.following variables may be collider : ".format(
                activate_var[0], activate_var[1]) + ','.join([node_names[i] for i in varlist])

        return "There don't exists collider between {} and {} ".format(activate_var[0], activate_var[1])
    except Exception as e:
        return f"tool raise error : {e}"


def Determine_confounder(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str)
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        filename = None
        cg = None
        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        rela, varlist = has_confounder(index_a, index_b, cg.G.graph)

        if rela == Relationship.YES:
            return "yes，There is an unblocked backdoor path between {} and {} so confounder exists. Backdoor path : ".format(
                varlist[0],
                activate_var[0]) + ','.join([node_names[i] for i in varlist])
        if rela == Relationship.UNCERTAIN:
            paths = ''
            for p in varlist:
                paths += "\n" + ','.join([node_names[i] for i in p])
            return "Whether there exist confounder between {} and {} is uncertain because some edge direction is uncertain.following paths may be backdoor path : ".format(
                activate_var[0], activate_var[1]) + paths
        return "no，No unblocked backdoor paths exist between {} and {}. So don't exist confounder".format(
            activate_var[0], activate_var[1])
    except Exception as e:
        return f"tool raise error : {e}"


def Determine_edge_direction(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str)
        filename = None
        cg = None

        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        a_to_b = cg.G.graph[index_a][index_b]
        b_to_a = cg.G.graph[index_b][index_a]
        prefix = ''
        if a_to_b == -1 and b_to_a == 1:
            return prefix + activate_var[0] + " is a directly cause of " + activate_var[
                1] + ".The opposite is not true"
        elif a_to_b == 1 and b_to_a == -1:
            return prefix + activate_var[1] + " is a cause of " + activate_var[
                0] + ".The opposite is not true"
        elif a_to_b == -1 and b_to_a == -1:
            return prefix + f"An undirected edge exists between  {activate_var[0]} and {activate_var[1]}, which means they are not independent. However, the direction of causality between the two variables is uncertain."
        else:
            return prefix + f"There is no direct edge linking  {activate_var[0]} and {activate_var[1]}.{activate_var[0]} doesn't directly cause {activate_var[1]}."
    except Exception as e:
        return str(e)


empty_tools = Tool(
    name='empty',
    func=empty,
    description='''if No action needed ,use this tool. use this tool default.input is original response\n'''
)
collider = Tool(
    name='Determine_collider',
    func=Determine_collider,
    description='''you should first generate causal graph and then use this tool.Useful When we are interested in whether there is a collider between two variables(ie common effect), we use this tool and the input is {"cg_name":...,"interesting_var":[...]}, where interesting_var is what Variable we want to test, cg_name is the name of causal generated by 'Generate Causal'.The output of the tool is yes or no or uncertainty and may be the variable name of the collider. Make sure the causal graph has been generated before using this tool\n'''
)
confound = Tool(
    name='Determine_confounder',
    func=Determine_confounder,
    description='''you should first generate causal graph and then use this tool.Useful When we are interested in whether there is a cofounder (ie common cause) between two variables, we use this tool and the input is {"cg_name":...,"interesting_var":[...]}, where interesting_var is what Variable we want to test, cg_name is the name of causal generated by 'Generate Causal'.The output of the tool is yes or no or uncertainty and the backdoor path that may lead to the existence of the cofounder. Make sure the causal graph has been generated before using this tool\n'''
)
edge_direction = Tool(
    name='Determine_edge_directions',
    func=Determine_edge_direction,
    description='''you should first generate causal graph and then use this tool.Useful when we are interested in whether there is a direct edge between two variables and the direction of the edge (such as determining whether A directly leads to B)., we use this tool and the input is {"cg_name"=...,"interesting_var"=[...]}, where interesting_var is what Variable we want to test, cg_name is the name of causal generated by 'Generate Causal'.The output of the tool is the relationship of two variables (ie A cause B). Make sure the causal graph has been generated before using this tool\n'''
)

condition_independent_test_tools = Tool(
    name='condition independent test',
    func=condition_independent_test,
    description='''Useful for when you need to test the *** independent or d-separate *** of variable A and variable B condition on variable C. input should be a json with format below {"filename":...,"interesting_var":[...],"condition":[...]},"interesting_var" is a list of variables user interested in. for example, if user want to test independent(d-separate) between X and Y condition on Z,W,Q , interesting_var is ["X","Y"], condition is ["Z","W","Q"]. condition is [] if no condition provided\n'''
)
generate_causalgraph_tool = Tool(
    name='Generate Causal',
    func=generate_causalgraph,
    description='''Useful for when you need to generate causal graph (or partial causal graph). input should be a json with format below {"filename":...,"analyse_relationship":...,"interesting_var":[...](Optional)}.if you want to analyse relationship between variables( such as cause effect, coufounder , Collider), analyse_relationship = "True" and please generate complete causal graph and  interesting_var is [](which means causal graph contain all variables) .if we only need to generate **partial causal graph** (for example, generate a partial causal graph for some variables), interesting_var is used and it's values are list of variables appear in causal graph and analyse_relationship is "False".Further more, if needed, you can analyse variables relationship in causal graph generated by this tool through these tools : Determine_collider,Determine_confounder,Determine_edge_direction\n'''
)
ate_tool = Tool(
    name='calculate CATE',
    func=ATE_sim,
    description='''Useful for when you need to calculate (conditional) average treatment effect (ATE or CATE, etc. in math function is E(Y(T=T1)-Y(T=T0) | X=x) and means if we use treatment, what uplift we will get from treatment).This tool use double machine learn algorithm to calculate ate. input is  a json with format {"filename":...,config: {Y:[...],T:[...],X:[...],T0:...,T1:...} }. Y are names of outcome, T are names of treatment, X are names of covariate affect both T and Y (i.e. confounder). T1 and T0 are two different values of T that need to be calculated in ATE. you should extract each name from the description.\n'''
)


llm = ChatOpenAI(temperature=0.5, openai_api_key=api_key,
                 model_name="gpt-3.5-turbo",openai_api_base='')

prompt = hub.pull("hwchase17/react")

ICL_lizi = """
##DEMO：
The doctor wants to explore the relationship between smoking, lung cancer, and yellow fingers, so he collects a batch of data, stores it in 'data.csv', and gives the following proposition: if all of the statements are false, output no; if one of the statements is true, output uncertain; if all of the statements are true, output yes. The proposition is as follows:

1. Yellow fingers are a cause of lung cancer
2. There is a collider between yellow finger and lung cancer
3. There is a confounder between yellow fingers and lung cancer
4. Yellow fingers and lung cancer are independent of each other given smoking

The output is just formatted as a json string, such as {{"answer":"yes"}}.

Thought: I need to generate a causal diagram, and then analyze the causal and independent relationships between variables to judge the correctness of each proposition
Action:Generate Causal
Action Input:{{"filename": "data.csv", "analyse_relationship": "True"}}
Observation:causal graph named 'data' is generate succeed!  and have written to the memory.

Thought: I need to use the Determine_edge_directions tool to determine if the yellow finger is the cause of smoking
Action:Determine_edge_directions
Action Input:{{"cg_name": "data", "interesting_var": ["yellow fingers", "lung cancer"]}}
Observation:There is no direct edge linking  yellow finger and cancer


Thought: There is no direct causal link between yellow finger and lung cancer. The proposition "yellow finger is the cause of lung cancer" is wrong. I need to continue to determine whether there is a collider between yellow finger and lung cancer
Action:Determine_collider
Action Input:{{"cg_name": "data", "interesting_var": ["yellow fingers", "lung cancer"]}}
Observation:There don't exists collider Smoking of yellow fingers and lung cancer

Thought: There is no collider between yellow finger and lung cancer, I need to continue to determine whether there is a confounder between yellow finger and lung cancer
Action:Determine_confounder
Action Input:{{"cg_name": "data", "interesting_var": ["yellow fingers", "lung cancer"]}}
Observation:yes, There is an unblocked backdoor path between yellow finger and lung cancer so confounder exists. Backdoor path: yellow fingers, smoking, lung cancer

Thought: There is an unblocked backdoor path between yellow finger and lung cancer, so there is a confounder, and it is smoking, I need to continue to determine whether yellow finger and lung cancer are independent of each other given smoking
Action:condition independent test
Action Input:{{"filename":"data.csv","interesting_var":["yellow fingers","lung cancer"],"condition":["somking"]}}
Observation:yellow fingers and lung cancer is independent under conditions: smoking

Thought:i now get the final answer. I have analyzed the four propositions, among which the confounder between yellow fingers and lung cancer is correct, and the others are all wrong, so the answer is uncertian

Final Answer:{{"answer":"uncertain"}}

##Requirement: 
Answer the following questions with examples:

"""
prompt.template = 'Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result_gpt_3.5 of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nCheck you output and make sure it conforms! Do not output an action and a final answer at the same time.\n\nBegin!\n\n'+ICL_lizi +'Question: {input}\nThought:{agent_scratchpad}'

# exit(0)
tools = [condition_independent_test_tools, generate_causalgraph_tool, empty_tools, ate_tool,collider,confound,edge_direction]
agent = create_react_agent(llm, [condition_independent_test_tools, generate_causalgraph_tool, empty_tools, ate_tool,collider,confound,edge_direction], prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors="Check you output and make sure it conforms! Do not output an action and a final answer at the same time.")
# print(agent.agent.llm_chain.prompt.template)

data_loader = Dataloader()
data_loader.read_data('DGP_description/dataset_gt.json')
correct = 0
summary = 0
# os.remove('./result_gpt_3.5.jsonl')

skip_num = 0
memory = CGMemory()
# data = pd.read_csv(os.path.join('DGP_description/data', "4_6_3.csv"), header=None)
# data.columns = ["age","weight","sleep time","cancer"]
# name_out_to_in = ["age","weight","sleep time","cancer"]

# resp = agent_executor.invoke({"input": "Consider four elements : age, weight, sleep time, cancer. With the advancement of age, individuals should pay closer attention to their weight and sleep duration, as these factors can significantly impact their overall health and the risk of developing cancer. Doctors are very interested in the relationship between these variables, and therefore, they have chosen to collect a set of data through experiments. Please assist the doctors in answering whether there is a collider of weight on cancer.if exist collider, please give it name. csv data store in ‘4_6_3.csv’ ."})
output_name = 'result_5_18_icl_medical.jsonl'
with open(output_name,'r') as f:
    skip_num = len(f.readlines())
print(f"{skip_num}")

for index in range(664):
    summary += 1
    line = data_loader.read_one(1)
    if index < skip_num:
        continue
    q, gt, name, col, q_type = data_loader.Get_question_answer_pair(line)
    data = pd.read_csv(os.path.join('DGP_description/data', name), header=None)
    data.columns = col
    name_out_to_in = col
    if q_type in ['IT', "CIT", "MULTCIT", "CAUSE", 'Has-Collider', 'Has-Confounder']:
        q = q + "\n Please output as json format {{'answer' : '...'}},answer should be one of ['yes','no','uncertain'],Do not output content other than JSON "
    elif q_type not in ["ATE"]:
        q = q + "\n Please only output the causal graph's name directly.output must be json format {{'answer' : '...'}},answer should be the name of the causal graph.you needn't really analyse relationship between variables."
    else:
        q = q + "\n Please only output the value of ate.output must be json format {{'ate' : '...'}}"

    print(q)
    memory = CGMemory()
    resp = agent_executor.invoke({"input": q})
    memory.clear()
    line['output'] = resp['output']
    print(f"{resp['output']}  gt: {gt}")

    if q_type != "ATE":
        if resp['output'].upper() == gt.upper():
            correct += 1
        else:
            print("wrong")
    else:
        if resp['output'] == gt:
            correct += 1
        else:
            print("wrong")
    if q_type in ["CAUSALKG", "PARTIAL_CG"]:
        with open(os.path.join('DGP_description/causal_graph', gt), "r") as f:
            with open("tempCG.txt", "r") as f2:
                cg_str = f2.read()
            if f.read() == cg_str:
                correct += 1
                line['match'] = "MATCH"
                print("MATCH")
            else:
                line['match'] = "MISMATCH"
                print("NOT MATCH")
    with open(f'{output_name}', 'a+') as f:
        json.dump(line, f, ensure_ascii=False)
        f.write('\n')

print(correct, summary)
