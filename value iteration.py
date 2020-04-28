'''
Please go to my Blog to read the detail of value iteration and policy iteration 
https://smiler666.github.io/post/rl-policy-iteration/
Welcome criticism and correction
'''
import copy
import numpy as np

states = ["1", "2"]
actions = ["a", "b"]
discount_factor = 0.99  # 奖励的折扣因子
# q_value 初始化
q_value = {states[0]: {actions[0]: 0, actions[1]: 0}, states[1]: {actions[0]: 0, actions[1]: 0}}  # 初始化Q-table
# 策略 pi 初始化
pi = {states[0]: {actions[0]: 0.5, actions[1]: 0.5}, states[1]: {actions[0]: 0.5, actions[1]: 0.5}}  # 初始化策略


def p_s_r(state, action):
    # 返回 跳转下一状态概率p，下一状态s_，和立即汇报
    if state == "1":
        if action == "a":
            return ((1.0 / 3, "1", 0),
                    (2.0 / 3, "2", 1))
        else:
            return ((2.0 / 3, "1", 0),
                    (1.0 / 3, "2", 1))
    if state == "2":
        if action == "a":
            return ((1.0 / 3, "1", 0),
                    (2.0 / 3, "2", 1))
        else:
            return ((2.0 / 3, "1", 0),
                    (1.0 / 3, "2", 1))


# finding optimal value function
def value_iteration():
    v_value = {states[0]: 0, states[1]: 0}
    threshold = 0.0001
    count = 0
    while True:
        count += 1
        v_value_old = copy.deepcopy(v_value)
        for s in states:
            temp_v = 0
            q_s_a = []
            for a, p in pi[s].items():
                temp_q = 0
                for t in p_s_r(s, a):
                    p_s_s1, s_, r = t[0], t[1], t[2]
                    temp_q += p_s_s1 * (r + discount_factor * v_value[s_])
                q_value[s][a] = temp_q
                temp_v += p * temp_q
                q_s_a.append(temp_v)
            # print(q_s_a)
            v_value[s] = max(q_s_a)

        # 收敛判断
        delta = 0
        for i in range(len(v_value)):
            delta += np.abs(v_value[states[i]] - v_value_old[states[i]])
        if delta <= threshold:
            print('Value-iteration converged at iteration# %d.' % count)
            break
        # print(v_value)
        # print(v_value_old)
    return v_value


# policy extraction
def policy_extraction(v):
    for s in states:
        action = max(pi[s], key=pi[s].get)
        for k in pi[s]:
            if k == action:
                if pi[s][k] != 1.0:
                    pi[s][k] = 1.0
            else:
                pi[s][k] = 0.0


if __name__ == '__main__':
    optimal_v = value_iteration()
    policy_extraction(optimal_v)
    print("状态值为：")
    print(optimal_v)
    print("行为值为：")
    print(q_value)
    print("策略为：")
    print(pi)
