# -*- coding: UTF-8 -*-
"""
@Date  : 2022/5/24 10:32 
@Author: Chongyan
@Aiming: Distribution Grid Topology Identification with LASSO, LS, RIDGE and other methods
"""

import pandas as pd
import numpy as np
import random as rnd
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import Lasso
import networkx as nx
from sklearn.linear_model import LassoCV

# 根据基尔霍夫定理构造数据集  I_i = sigma(V_i - V_j) * 1/r_ij
# N节点， M个时间点数据, NxN 维的r阻抗数据；记录M个时间点的 各节点的电流 电压数据， 反推 r_ij
# V_i = Vhat* sin(w*t + w_i); Vhat=220, w=50, t = 1,2,...,M, w_i 随机选取[0,2*pi]
# r_ij 得确保连通性 和 稀疏性

cid = rnd.seed(1)

def generate_data(m):
    # 利用 networkx的树图 保障r_ij的连通性
    # vhat, w = 220, 50
    # wi = np.random.uniform(low=0, high=6, size=(1, n))
    # t = np.array(list(range(1, m+1)))
    # G = nx.random_powerlaw_tree(n=n, gamma=3, seed=cid, tries=10000)
    #G = nx.connected_watts_strogatz_graph(n=n, k=6, p=0.5, tries=100)
    btree = nx.balanced_tree(10, 2)
    G = nx.disjoint_union(btree, btree)
    n = len(G.nodes)

    print(nx.is_connected(G))
    # nx.draw(G)
    # plt.show()
    adj_mat = nx.adjacency_matrix(G)
    r = sparse.csr_matrix(np.multiply(adj_mat.todense(), np.random.uniform(low=2, high=4, size=(n, n))))
    #v = vhat * np.sin(w * t.reshape(m, 1).repeat(n, axis=1) + wi.repeat(m, axis=0))
    v = np.random.normal(loc=220, scale=10, size=(m, n))

    Iflow = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            neighb = G.neighbors(j)
            for item in neighb:
                Iflow[i, j] += (v[i, j]-v[i, item])/r[j, item]
    np.savetxt(f"data/Disjoint_adj_m{m}.csv", adj_mat.todense(), delimiter=",", fmt='%d')
    np.savetxt(f"data/Disjoint_r_m{m}.csv", r.todense(), delimiter=",", fmt='%.2f')
    np.savetxt(f"data/Disjoint_v_m{m}.csv", v, delimiter=",", fmt='%.2f')
    np.savetxt(f"data/Disjoint_iflow_m{m}.csv", Iflow, delimiter=",", fmt='%.2f')
    return adj_mat, r, v, Iflow


def graph_connected():
    adj = pd.read_csv(f'data/Disjoint_adj_m{m}.csv', header=None)
    adj = np.matrix(adj)
    G = nx.from_numpy_matrix(adj)
    return nx.is_connected(G)


def dataset_nodei(nodeIndex, m, n, percent):
    # 按节点构造 y 和 Theta 矩阵 y|mx1 = Ii|mx1, Theta|mxn, Theta_i[m,j] = v[m,i]-v[m,j], j∈[1,n]. xij=1/rij
    # 注意这里的下标i是固定的，指代节点i，这一次的拓扑辨识都是针对节点i的，xij!=0，则说明i与节点j有关联。xij是要回归的系数
    v_mn = pd.read_csv(f'data/Disjoint_v_m{m}.csv', header=None)
    dfx = v_mn.sub(v_mn.iloc[:, nodeIndex], axis=0) * -1
    dfx.drop(columns=[nodeIndex], axis=1, inplace=True)
    i_mn = pd.read_csv(f'data/Disjoint_iflow_m{m}.csv', header=None)
    dfy = i_mn.iloc[:, nodeIndex]
    mask = rnd.sample(range(m), int(m*percent))
    dfx = dfx.iloc[mask, :]
    dfy = dfy.iloc[mask]

    return dfx, dfy


def TI_lasso(pct, nodeIndex, m, n):

    X, y = dataset_nodei(nodeIndex=nodeIndex, m=m, n=n, percent=pct)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=10)
    model = LassoCV(cv=5, random_state=10, max_iter=10000)
    model.fit(X_train, y_train)
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)
    return list(zip(lasso_best.coef_, X))


# 电压相关性检测
def volt_correlation(nodeIndex,m,n):
    v_mn = pd.read_csv(f'data/Disjoint_v_m{m}.csv', header=None)
    dfx = v_mn.sub(v_mn.iloc[:, nodeIndex], axis=0) * -1
    dfx.drop(columns=[nodeIndex], axis=1, inplace=True)
    base = dfx[dfx.columns[0]]
    x, y = [], []
    for item in range(1, dfx.shape[1]):
        x.append(dfx.columns[item])
        y.append(base.corr(dfx[dfx.columns[item]]))
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel("Node Label")
        plt.ylabel("Pearson Correlation")
        plt.show()

    return None



def classify_result(pct, m, n):
    xaxis, yaxis, color = [], [], []
    for node in range(n):
        coef = TI_lasso(pct, node, m, n)
        adj = pd.read_csv(f'data/Disjoint_adj_m{m}.csv', header=None)
        for item in coef:
            xaxis.append(node)
            yaxis.append(item[0])
            if adj.iloc[node, item[1]] == 1:
                color.append('r')
            else:
                color.append('g')
    plt.scatter(xaxis, yaxis, c=color, alpha=0.5)
    plt.xlabel(f"Node Label with {100*pct}% Measurements ")
    plt.ylabel("Coefficient")
    plt.show()



m = 300
pct = 0.5
#volt_correlation(3,m,n)
adj, r, v, iflow = generate_data(m)
n = adj.shape[0]
classify_result(pct, m, n)
#gtree = nx.balanced_tree(4, 3)