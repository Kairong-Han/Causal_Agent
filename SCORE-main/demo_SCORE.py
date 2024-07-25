import random

from stein import *
import numpy as np
import cdt
import os

def generate(d, s0, N, noise_std = 1, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
    X, noise_var = teacher.sample(N)
    return X, adjacency

# Data generation paramters
graph_type = 'ER'
d = 10
s0 = 10
N = 1000


X, adj = generate(d, s0, N)
node_num = [i for i in range(3,11)]

def edge_num(node_num):
    max_num = int(node_num*(node_num-1)/2)
    if node_num:
        return [i for i in range(0,max_num+1)]

folder = 'data'
file_to_save = "{}_{}_{}.csv"
# gt_to_save = "gtnode_{}_edge_{}.csv"
for i in node_num:
    for ind in range(20):
        edge = random.choice(edge_num(i))
        X, adj = generate(i, edge, 1000)
        numpy_array = X.numpy()
        np.savetxt(os.path.join(folder,file_to_save.format(i,edge,ind)), numpy_array, delimiter=',')
        # np.savetxt(os.path.join(folder,gt_to_save.format(i,edge)), adj, delimiter=',')
        print(i,ind)

# SCORE hyper-parameters
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001

A_SCORE, top_order_SCORE =  SCORE(X, eta_G, eta_H, cam_cutoff)
# print("SHD : {}".format(SHD(A_SCORE, adj)))
# print("SID: {}".format(int(cdt.metrics.SID(target=adj, pred=A_SCORE))))
# print("top order errors: {}".format(num_errors(top_order_SCORE, adj)))
