import numpy as np
def mutilabel_acc(output, target):
    D = len(output)
    temp = 0
    for i in range(D):
        jiaoji = np.sum(output[i] == target[i])
        bingji = np.sum(output[i] == 1) + np.sum(target[i] == 1)
        temp += jiaoji / bingji
    return temp / D

def mutilabel_pre(output,target):
    D = len(output)
    temp = 0
    for i in range(D):
        jiaoji = np.sum(output[i] == target[i])
        Z = np.sum(output[i] == 1)
        temp += jiaoji / Z
    return temp / D

def mutilabel_recall(output,target):
    D = len(output)
    temp = 0
    for i in range(D):
        jiaoji = np.sum(output[i] == target[i])
        F = np.sum(target[i] == 1)
        temp += jiaoji / F
    return temp / D

def mutilabel_Fvalue(output,target):
    D = len(output)
    temp = 0
    for i in range(D):
        jiaoji = np.sum(output[i] == target[i])
        Z = np.sum(output[i] == 1)
        F = np.sum(target[i] == 1)
        temp += 2 * jiaoji / (Z + F)
    return temp / D





if __name__ == "__main__":
    out = [np.array([1,0,0])]
    target = [np.array([1,0,0])]
    print(mutilabel_acc(out,target))