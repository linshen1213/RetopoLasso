# -*- coding: UTF-8 -*-
"""
@Date  : 2022/8/23 14:19 
@Author: Chongyan
@Aiming:
"""
import pandas as pd
import numpy as np
import os
import sys
import pandapower as pp


def random_load(times, busNum):
    v = np.random.normal(loc=1, scale=1, size=(times, busNum))
    # v = np.where(v > 0, v, -v) # 把负load改为正的
    return v


def build_case69(df, rnd_load, timeN):
    net = pp.create_empty_network()
    # build 69+1 buses
    # df = pd.read_csv(f'IEEE69.csv')
    for i in range(70):
        if i == 0:
            pp.create_bus(net, vn_kv=33)
        else:
            pp.create_bus(net, vn_kv=11)
    pp.create_ext_grid(net, bus=0, vm_pu=1.02)
    # create lines based on the df, and add loads to the buses
    for index, row in df.iterrows():

        pp.create_line_from_parameters(net, from_bus=int(row['from']), to_bus=int(row['to']), length_km=1,
                                       r_ohm_per_km=float(row['rohm']), x_ohm_per_km=float(row['xohm']), c_nf_per_km=0,
                                       max_i_ka=float(row['maxi']) / 100.0)
        # pp.create_load(net, bus=int(row['to']), p_mw=P_incr+float(row['P'])/1000, q_mvar=Q_incr+float(row['Q'])/1000)
    for i in range(rnd_load.shape[1]):
        pp.create_load(net, bus=i+1, p_mw=rnd_load[timeN][i])
    # add the transformer
    pp.create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, sn_mva=6.0, vn_hv_kv=33.0, vn_lv_kv=11.0,
                                          vkr_percent=0.25, vk_percent=12.0, pfe_kw=30, i0_percent=0.06)
    return net


def power_thru_bus(df, rnd_load, timeN):

    net = build_case69(df, rnd_load, timeN)
    pp.rundcpp(net)
    resBus = net.res_bus
    resLine = net.res_line
    bus_to_ID = {}   # 记录 {fromBus:[(toBus,busID),...]}
    bus_load = {}
    for i in range(df.shape[0]):
        from_bus = int(df.iloc[i]['from'])
        to_bus = int(df.iloc[i]['to'])
        # load = P_incr + float(df.iloc[i]['P'])/1000
        bus_to_ID.setdefault(from_bus, [])
        bus_to_ID[from_bus].append((to_bus, i))
        # bus_load[to_bus] = load
    bus_P = {}  # 计算一个断面中各个bus的功率和相角
    bus_Va = {}
    for i in range(1, resBus.shape[0]):
        pi = 0
        if i in bus_to_ID:
            pi += sum([float(resLine.iloc[j[1]]['p_from_mw']) for j in bus_to_ID[i]])
        pi += rnd_load[timeN][i-1]
        bus_P[i] = round(pi, 6)
        bus_Va[i] = round(float(resBus.iloc[i]['va_degree']), 6)

    return bus_P, bus_Va



def generate_69adj():
    df = pd.read_csv(f'IEEE69.csv')
    adj = [[0 for i in range(69)] for j in range(69)]
    for i in range(df.shape[0]):  # 注意下标与编号的-1关系
        from_bus = int(df.iloc[i]['from'])-1
        to_bus = int(df.iloc[i]['to'])-1
        adj[from_bus][to_bus], adj[to_bus][from_bus] = 1, 1
    pd.DataFrame(adj).to_csv('data/69_adj.csv', index=False, header=0)


def generate_P_Va():
    df = pd.read_csv(f'IEEE69.csv')
    times = 200
    busNum = 69
    rndload = random_load(times, busNum)
    P_set = []
    Va_set = []
    for i in range(times):
        print(i,'/',times)
        # P_incr = i/1000.0
        P, Va = power_thru_bus(df, rndload, i)
        P_set.append(P.values())
        Va_set.append(Va.values())
    pd.DataFrame(P_set).to_csv('data/dcpf_69_P.csv', index=False, header=0)
    pd.DataFrame(Va_set).to_csv('data/dcpf_69_Va.csv', index=False, header=0)
# generate_69adj()
generate_P_Va()
print('done')





