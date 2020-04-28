'''
Please go to my Blog to read the detail of value iteration and policy iteration 
https://smiler666.github.io/post/rl-policy-iteration/
Welcome criticism and correction
'''

import numpy as np
import copy

states = ['1', '2']
actions = ['a', 'b']
rewards = [0, 1]
discount_factor = 0.9  # 奖励折扣因子
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


def policy_evaluate():
    v_value = {states[0]: 0, states[1]: 0}
    threshold = 0.0001
    while True:
        v_value_old = copy.deepcopy(v_value)

        for s in states:
            temp_v = 0
            # temp_q = 0
            for a, p in pi[s].items():
                temp_q = 0
                for t in p_s_r(s, a):
                    p_s_s1, s_, r = t[0], t[1], t[2]
                    temp_q += p_s_s1 * (r + discount_factor * v_value[s_])
                q_value[s][a] = temp_q
                temp_v += p * temp_q
            v_value[s] = temp_v

        # 收敛判断
        delta = 0
        for i in range(len(v_value)):
            delta += np.abs(v_value[states[i]] - v_value_old[states[i]])
        if delta <= threshold:
            break
    # print(v_value)
    # print(v_value_old)
    return v_value


def policy_improve(v):
    done = True

    for s in states:
        action = max(q_value[s],key=q_value[s].get)
    # print(action)
        for k in pi[s]:
            if k == action:
                if pi[s][k] != 1.0:
                    pi[s][k] = 1.0
                    done = False
            else:
                pi[s][k] = 0.0
    return done


if __name__ == '__main__':
    is_done = False
    i = 0
    while is_done is False:
        v = policy_evaluate()
        is_done = policy_improve(v)
        i += 1

    print('Policy-Iteration converged at step %d.' % i)
    print("状态值为：")
    print(v)
    print("行为值为：")
    print(q_value)
    print("策略为：")
    print(pi)