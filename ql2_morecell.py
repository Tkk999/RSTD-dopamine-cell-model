import time
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib

N_STATES = 6
ACTIONS = ["fixed", "risky"]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 15
FRESH_TIME = 0.3
TerminalFlag = 119
ESTIMATE = 445
matplotlib.rcParams.update({'font.size': 12})
# plt.xlim(100,800)

class Cell():
    def __init__(self, lr_p, lr_n, q_table,q_list):
        self.lr_p = lr_p
        self.lr_n = lr_n
        self.q_table = q_table
        self.q_list = q_list

    def get_q_table(self):
        return self.q_table


def build_q_table(n_states, actions):
    return pd.DataFrame(
        np.full((n_states, len(actions)),445),
        columns=actions
    )


def estamate_action(state, q_table, state_type):
    state_table = q_table.loc[state, :]
    index = state
    next_state_type = state_type.iloc[index + 1]
    recall = []
    while index != 0 and len(recall) < 5:
        if state_type[index] == next_state_type:
            recall.append(state_type[index])
        index = index - 1

    q_table.loc[state, "risky"] = np.mean(recall)


def choose_action(state, cell_list):
    # if state == 0:
    #
    # else:
    temperture = 2
    q_list_fix = []
    q_list_risk = []
    for cell in cell_list:
        # print("222222222222222222222222")
        # print(cell.q_table["fixed"].iloc[state])
        # print(cell.q_table["risky"].iloc[state])
        q_list_fix.append(cell.q_table["fixed"].iloc[state])
        q_list_risk.append(cell.q_table["risky"].iloc[state])

    avg_fix = np.mean(q_list_fix)
    # print(avg_fix)
    avg_risk = np.mean(q_list_risk)
    # print(avg_risk)

    p_fix = np.exp(avg_fix/temperture) / (np.exp(avg_fix/temperture) + np.exp(avg_risk/temperture))
    # print(p_fix)

    if p_fix > 0.5:
        action = "fixed"
    else:
        action = "risky"

    return action


def update_env(S, A):

    if S == 1:
        if A == "fixed":
            output = random.randint(200, 290)
        else:
            temp = random.randint(0, 1)
            if temp == 0:
                output = random.randint(100, 190)
            else:
                output = random.randint(300, 390)
    else:
        if A == "fixed":
            output = random.randint(600, 690)
        else:
            temp = random.randint(0, 1)
            if temp == 0:
                output = random.randint(500, 590)
            else:
                output = random.randint(700, 790)

    temp = random.randint(0, 5)
    if temp < 3:
        S_ = 0
    else:
        S_ = 1

    return S_, output


def rl(df):

    q_table1 = build_q_table(2, ACTIONS)
    q_table2 = build_q_table(2, ACTIONS)
    q_table3 = build_q_table(2, ACTIONS)
    q_table4 = build_q_table(2, ACTIONS)
    q_table5 = build_q_table(2, ACTIONS)
    q_table6 = build_q_table(2, ACTIONS)
    q_table7 = build_q_table(2, ACTIONS)
    q_table8 = build_q_table(2, ACTIONS)
    q_table9 = build_q_table(2, ACTIONS)

    a1_q = []
    a2_q = []
    a3_q = []
    a4_q = []
    a5_q = []
    a6_q = []
    a7_q = []
    a8_q = []
    a9_q = []

    a1 = Cell(0.6, 0.1, q_table1, a1_q)
    a2 = Cell(0.5, 0.4, q_table2, a2_q)
    a3 = Cell(0.6, 0.2, q_table3, a3_q)
    a4 = Cell(0.3, 0.3, q_table4, a4_q)
    a5 = Cell(0.4, 0.4, q_table5, a5_q)
    a6 = Cell(0.5, 0.5, q_table6, a6_q)
    a7 = Cell(0.4, 0.5, q_table7, a7_q)
    a8 = Cell(0.3, 0.4, q_table8, a8_q)
    a9 = Cell(0.1, 0.6, q_table9, a9_q)

    cell_list = [a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # cell_list = [a7, a8, a9]
    # cell_list = [a1, a2, a3]
    # cell_list = [a4, a5, a6]
    q_table_list = []
    high_middle = []
    high_end = []
    low_middle = []
    low_end = []

    reward_distribution = []
    reward_distribution_high = []
    reward_distribution_low = []

    action_list = []
    reward_list = []

    # c4 = np.random.randint(0, 50, 1000)

    temp = random.randint(0, 5)
    if temp < 3:
        S = 0
    else:
        S = 1

    for i in range(0, 1000):
        # print("-----------------------",i)

        A = choose_action(S,cell_list)
        # print(A)
        if A == "risky":
            action_list.append(1)
        else:
            action_list.append(0)

        S_, R = update_env(S, A)
        reward_list.append(R)

        new_cell_list = []

        for cell in cell_list:
            if A == "risky":
                q_predict = cell.q_table["risky"].iloc[S]
                q_target = R + GAMMA * cell.q_table["risky"].iloc[S_]
                RPE = R - q_predict
                if RPE > 0:
                    cell.q_table.loc[S, "risky"] += cell.lr_p * (R - q_predict)  # expectiles (learning rate * RPE)
                else:
                    cell.q_table.loc[S, "risky"] += cell.lr_n * (R - q_predict)
                cell.q_list.append(cell.q_table.loc[S, "risky"])
            else:
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11111111111111111111111111111111")
                q_predict_fix = cell.q_table["fixed"].iloc[S]
                q_target_fix = R + GAMMA * cell.q_table["fixed"].iloc[S_]
                RPE = R - q_predict_fix
                if RPE > 0:
                    cell.q_table.loc[S, "fixed"] += cell.lr_p * (R - q_predict_fix)  # expectiles (learning rate * RPE)
                else:
                    cell.q_table.loc[S, "fixed"] += cell.lr_n * (R - q_predict_fix)
                # print(cell.q_table.loc[S, "fixed"])
                cell.q_list.append(cell.q_table.loc[S, "fixed"])

            new_cell_list.append(cell)

            if i == 500:
                low_middle.append(cell.q_table["risky"].iloc[1])
                high_middle.append(cell.q_table["risky"].iloc[0])

            if i == 999:
                low_end.append(cell.q_table["risky"].iloc[1])
                high_end.append(cell.q_table["risky"].iloc[0])

        cell_list = new_cell_list

        S = S_

    return low_middle,low_end,high_middle,high_end,reward_list, cell_list,action_list



if __name__ == '__main__':
    data = pd.read_csv('File 1.csv')
    df = data[data['trialType'].str.contains('L|H')]
    of = df['outcomeVal']
    choice = df['choice.clicked_name']

    low_mid = [0] * 1000
    low_end = []
    high_mid = [0] * 1000
    high_end = []
    reward_list = []
    cell_list_new = []
    action_list_new = []


    for i in range(0,10):
        low_m, low_e, high_m, high_e, reward, cell_list, action_list = rl(df)
        low_mid = np.sum([low_mid,low_m], axis=0)
        low_end = np.sum([low_end,low_e], axis=0)
        high_mid = np.sum([high_mid,high_m], axis=0)
        high_end = np.sum([high_end,high_e], axis=0)
        reward_list = reward_list + reward
        action_list_new = action_list_new + action_list
        cell_list_new.append(cell_list)

    # print(low_end)
    # print(high_end)
    low_end = np.array(low_end)
    high_end = np.array(high_end)
    low_end = low_end / 10
    high_end = high_end / 10

    reward_list = np.array(reward_list)
    reward_list = reward_list
    # print(reward_list)

    sns.distplot(reward_list)
    plt.show()
    # sns.distplot(action_list_new)

    # df = pd.DataFrame({'a': range(len(cell_list[0].q_list)), 'b': cell_list[0].q_list})
    # sns.relplot(x="a", y="b", kind="line", data=df)
    #
    # print(cell_list[0].q_list)
    # print(cell_list[1].q_list)
    # print(cell_list[2].q_list)
    # print(cell_list[3].q_list)
    # print(cell_list[4].q_list)
    #
    # fig = plt.figure(figsize=(12, 5))
    #
    # ax = plt.subplot(331)
    # df = pd.DataFrame({'a': range(len(cell_list[0].q_list)), 'b': cell_list[0].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df,ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.title("1")
    #
    # ax = plt.subplot(332)
    # df = pd.DataFrame({'a': range(len(cell_list[1].q_list)), 'b': cell_list[1].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    #
    # plt.title("1")
    # ax = plt.subplot(333)
    # df = pd.DataFrame({'a': range(len(cell_list[2].q_list)), 'b': cell_list[2].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.title("1")
    # ax = plt.subplot(334)
    # df = pd.DataFrame({'a': range(len(cell_list[3].q_list)), 'b': cell_list[3].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.title("1")
    #
    # ax = plt.subplot(335)
    # df = pd.DataFrame({'a': range(len(cell_list[4].q_list)), 'b': cell_list[4].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    #
    # plt.title("1")
    #
    # ax = plt.subplot(336)
    # df = pd.DataFrame({'a': range(len(cell_list[5].q_list)), 'b': cell_list[5].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.title("1")
    # ax = plt.subplot(337)
    # df = pd.DataFrame({'a': range(len(cell_list[6].q_list)), 'b': cell_list[6].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.title("1")
    # ax = plt.subplot(338)
    # df = pd.DataFrame({'a': range(len(cell_list[7].q_list)), 'b': cell_list[7].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.title("1")
    # ax = plt.subplot(339)
    # df = pd.DataFrame({'a': range(len(cell_list[8].q_list)), 'b': cell_list[8].q_list})
    # # sns.relplot(x="a", y="b", kind="line", data=df, ax = ax)
    # plt.plot(range(len(cell_list[0].q_list)), cell_list[0].q_list)
    # plt.show()

    sns.distplot(low_end, bins=10, hist=True, kde=True, rug=False, norm_hist=False, color='y', label='predict reward',
                 axlabel='reward')
    plt.suptitle("No preference low")
    plt.xlim(100, 400)
    plt.show()

    print(high_end)

    sns.distplot(high_end, bins=10, hist=True, kde=True, rug=False, norm_hist=False, color='y', label='predict reward',
                 axlabel='reward')
    plt.suptitle("No preference high")
    plt.xlim(600, 800)
    plt.show()



    # # Low
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = plt.subplot(121)
    # rs = np.random.RandomState(10)  # 设定随机数种子
    # # s = pd.Series(rs.randn(100) * 100)
    # # s = np.random.randint(0, 50, 1000)
    # sns.distplot(low_end, bins=10, hist=True, kde=True, rug=False,norm_hist=False, color='y', label='predict reward', axlabel='reward')
    # plt.suptitle("Low compare High")
    # # plt.xlim(100,800)
    # plt.legend()
    #
    # ax1 = plt.subplot(122)
    # sns.distplot(high_end, bins=10, hist=True, kde=True, rug=False, norm_hist=False, color='y', label='predict reward',axlabel='reward')
    # # plt.xlim(100,800)
    # # plt.suptitle("High Middle & End")
    # # plt.legend()
    #
    # # sns.kdeplot(rew, shade=True, color='g')
    #
    # # sns.distplot(rew, bins=10, hist=True, kde=True, rug=True,shade=True, norm_hist=False, color='g', label='distplot', axlabel='reward')
    #
    # plt.show()



    # # High
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = plt.subplot(121)
    # sns.distplot(high_m, bins=10, hist=True, kde=True, rug=False,norm_hist=False, color='y', label='predict reward', axlabel='reward')
    # plt.suptitle("High Middle compare End")
    # plt.legend()
    #
    # ax1 = plt.subplot(122)
    # sns.distplot(high_e, bins=10, hist=True, kde=True, rug=False, norm_hist=False, color='y', label='predict reward',axlabel='reward')
    # plt.show()
    #
    #
    #
    # plt.show()

