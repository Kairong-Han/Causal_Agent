import os
import pandas as pd
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.FCMBased.PNL.PNL import PNL

# default parameters


folder = "../SCORE-main/data"
def G_to_DAG(adj,data):
    for i in range(len(adj)-1):
        for j in range(i+1, len(adj)):  # 从i到N，因为j>=i时才需要检查
            if adj[i][j] == -1 and adj[j][i] == -1:
                pnl = PNL()
                p_value_foward, p_value_backward = pnl.cause_or_effect(data[:,i], data[:,j])
                if p_value_foward < p_value_backward: #i->j
                    adj[j][i] = 1
                else:
                    adj[i][j] = 1

    return adj

for file in os.listdir(folder):
    df = pd.read_csv(os.path.join(folder,file))
    df = np.array(df)
    cg = pc(df)
    print(cg.G.graph)
    cg.draw_pydot_graph()
    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png(os.path.join("CG",f'{file}.png'))
