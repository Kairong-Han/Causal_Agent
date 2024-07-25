import json

import numpy as np
import pandas as pd
from enum import Enum

from typing import Tuple


class Relationship(Enum):
    YES = 1
    NO = 0
    UNCERTAIN = 2

class Edge(Enum):
    A2B = 1
    B2A = 2
    NOARROW = 0
    ARROW = 3
    NOEDGE = 4
def triangle_to_adj(triangle):
    adj = {}
    for i in range(len(triangle)):
        adj[i] = [j for j, value in enumerate(triangle[i]) if value != 0]
    return adj
def find_all_paths(i, j, adj, path=[]):
    path = path + [i]
    if i == j:
        return [path]
    if i not in adj:
        return []
    paths = []
    for node in adj[i]:
        if node not in path:
            newpaths = find_all_paths(node, j, adj, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def edge_dir(X:int,Y:int,adj1) -> Edge:
    if adj1[X][Y] == -1 and adj1[Y][X] == 1:
        return Edge.A2B
    elif adj1[Y][X] == -1 and adj1[X][Y] == 1:
        return Edge.B2A
    elif adj1[Y][X] == -1 and adj1[X][Y] == -1:
        return Edge.NOARROW
    elif adj1[Y][X] == 1 and adj1[X][Y] == 1:
        return Edge.ARROW
    else:
        return Edge.NOEDGE

class state_machine():
    s = 0
    record = -1
    def Get_state(self):
        return self.s

    def Get_output(self):
        if self.s in [ 1, 5]:
            return Relationship.UNCERTAIN
        if self.s in [3,2,0]:
            return  Relationship.NO
        if self.s == 4:
            return Relationship.YES
        else:
            return "error"

    def Eat(self,edge):
        if edge == Edge.A2B:
            if self.s == 0:
                self.s = 3
            elif self.s == 1:
                self.s = 5
            elif self.s == 2:
                self.s = 4
            elif self.s == 3:
                self.s = 3
            elif self.s == 4:
                self.s = 4
            elif self.s == 5:
                self.s = 5
        elif edge == Edge.B2A:
            if self.s == 0:
                self.s = 2
            elif self.s == 1:
                self.s = 1
            elif self.s == 2:
                self.s = 2
            elif self.s == 3:
                self.s = 3
            elif self.s == 4:
                self.s = 3
            elif self.s == 5:
                self.s = 3
        elif edge == Edge.NOARROW:
            if self.s == 0:
                self.s = 1
            elif self.s == 1:
                self.s = 1
            elif self.s == 2:
                self.s = 1
            elif self.s == 3:
                self.s = 3
            elif self.s == 4:
                self.s = 5
            elif self.s == 5:
                self.s = 5
        else:
            self.s = 7

def has_confounder_path(path : list,adj1) -> Relationship:
    if len(path)<=2:
        return Relationship.NO
    sm = state_machine()
    for ind in range(len(path[:-1])):
        edge = edge_dir(path[ind],path[ind+1],adj1)
        sm.Eat(edge)
        if sm.Get_state() == 3:
            return sm.Get_output()
    return sm.Get_output()

def has_confounder(X : int ,Y : int ,adj1 : np.ndarray) -> Tuple[Relationship, list]:
    adj = triangle_to_adj(adj1)
    print(adj)
    paths = find_all_paths(X, Y, adj)
    ret_rela = Relationship.NO
    ret_path = []
    for path in paths: # 只要有一条路径有confounder就返回True
        rela= has_confounder_path(path,adj1)
        if rela == Relationship.YES:
            return rela,path
        elif rela == Relationship.UNCERTAIN:
            ret_rela = rela
            ret_path.append(path)
    return ret_rela,ret_path

def judge_col(X : int ,Y : int ,C:int,adj1 : np.ndarray) -> Relationship:
    a_to_c = adj1[X][C]
    c_to_a = adj1[C][X]
    b_to_c = adj1[Y][C]
    c_to_b = adj1[C][Y]
    if a_to_c == -1 and c_to_a == 1 and b_to_c == -1 and c_to_b == 1:
        return Relationship.YES
    if a_to_c == 1 and c_to_a == -1 or b_to_c == 1 and c_to_b == -1:
        return Relationship.NO
    return Relationship.UNCERTAIN
def has_collider(X : int ,Y : int ,adj1 : np.ndarray) -> Tuple[Relationship, list]:
    a_list_1 = np.nonzero(adj1[X][:])
    b_list_1 = np.nonzero(adj1[Y][:])
    inter = np.intersect1d(a_list_1, b_list_1)
    ret_rela = Relationship.NO
    ret_int = []
    for ind in inter:
        tmp = judge_col(X, Y, ind, adj1)
        if tmp == Relationship.YES:
            return Relationship.YES,[ind]
        if tmp == Relationship.UNCERTAIN:
            ret_rela = Relationship.UNCERTAIN
            ret_int.append(ind)
    if not ret_int:
        return Relationship.NO,[]
    return ret_rela,ret_int

# data = np.array(pd.read_csv('./tmp.csv',header=None))

class Dataloader():
    dataset = []
    index = 0
    def read_data(self,path):
        with open(path,'r') as f:
            for line in f:
                self.dataset.append(json.loads(line))
    def read_one(self,offset = 1):
        item = self.dataset[self.index]
        self.index += offset
        return item

    def Get_question_answer_pair(self,line):
        q = ''
        q = line['text'] + f"\nfile_name: {line['file']}\n Command: {line['Q']}"
        return q,line['gt'],line['file'],line['variables'],line['question_type']

class CGMemory():
    def __init__(self):
        self.CG = {}
        self.index = 1
    def get(self,name) -> tuple[object,Relationship]:
        if name not in self.CG.keys():
            return f"causal graph named {name} don't exist.please generate it at first.",Relationship.NO
        else:
            return self.CG[name],Relationship.YES
    def add(self,name,cg) -> tuple[str,Relationship]:
        if name in self.CG.keys():
            return f"causal graph named {name} already exist.please change name and retry.",Relationship.NO
        else:
            self.CG[name] = cg
            return "succeed",Relationship.YES
    def Delete(self,name)-> tuple[str,Relationship]:
        if name in self.CG.keys():
            del self.CG[name]
            return f"delete {name} successful",Relationship.YES
        else:
            return f"causal graph named {name} already exist.please change name and retry.",Relationship.NO
    def clear(self):
        self.CG = {}
    def Get_name(self,ind):
        self.index += 1
        return ind+"_"+str(self.index)