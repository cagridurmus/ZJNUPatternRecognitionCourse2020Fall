from numpy import *
import matplotlib.pyplot as plt


def load_file(filename, spliter='\t'):
    '''
    载入数据
    '''
    data = []
    with open(filename) as fr:
        for line in fr.readlines():
            line = line.strip().split(spliter)
            line = list(map(float, line))
            data.append(line)
    return data


def dist(a, b):
    '''
    计算二维平面欧式距离
    '''
    return sqrt(sum(power(a - b, 2)))


def rand_cent(data, k):
    '''
    对数据生成在范围内的k个随机质心
    '''
    n = shape(data)[1]
    ct = mat(zeros((k, n)))
    for j in range(n):
        min_j, max_j = min(data[:, j]), max(data[:, j])
        delta = float(max_j - min_j)
        ct[:, j] = min_j + delta * random.rand(k, 1)
    return ct


def c_means(dt, k):
    '''
    输入（坐标数据，划分数量）
    返回（质心，划分方案）
    '''
    m = shape(dt)[0]            # 数据个数
    ca = mat(zeros((m, 1)))     # 构造划分方案矩阵，每行一个元素，代表对应的点属于那个簇
    ct = rand_cent(dt, k)       # 构造质心矩阵，k行每行两个元素，代表坐标。初始状态随机选择质心。
    flag = True                 # 循环标志

    while flag:
        flag = False
        for j in range(m):  # 遍历数据中所有坐标
            # 定义最近的质心编号与最近质心的距离。
            min_i, min_d = 0, dist(ct[0, :], dt[j, :])

            # 遍历所有质心，找到最近的质心编号
            for i in range(1, k):
                d = dist(ct[i, :], dt[j, :])
                if d < min_d:
                    min_i, min_d = i, d

            # 如果和旧的划分方案不同，则更新当前点的划分方案，并激活循环标志
            if ca[j, 0] != min_i:
                ca[j, 0] = min_i
                flag = True

        # 根据本轮的划分方案，重新计算每个簇的质心
        for cent in range(k):
            p = dt[nonzero(ca == cent)[0]]
            ct[cent, :] = mean(p, axis=0)

    # 返回结果
    return ct, ca


def make_plot(data, ct, ca):
    '''
    绘图函数，输入（坐标数据，质心，划分方案）
    绘制过程不作为本次实验的重点
    '''
    m = shape(ct)[0]
    fig = plt.figure(figsize=(8, 7), dpi=250)
    markers = ['s', 'o', '^', 'p', 'd', 'v', 'h', '>', '<']
    colors = ['blue', 'green', 'purple', 'orange', 'black', 'brown']
    ax = fig.add_subplot(111)

    for i in range(m):
        p = data[nonzero(ca == i)[0], :]
        ax.scatter(
            p[:, 0].flatten().A[0],
            p[:, 1].flatten().A[0],
            marker=markers[i % len(markers)],
            c=colors[i % len(colors)],
            s=18,
            alpha=0.5)

    ax.scatter(
        ct[:, 0].flatten().A[0],
        ct[:, 1].flatten().A[0],
        marker='+',
        c='red',
        s=200)


def main(filename, c):
    dataset = mat(load_file(filename, spliter=','))
    centroid, assignment = c_means(dataset, c)
    print(centroid)
    make_plot(dataset, centroid, assignment)
    plt.show()


if __name__ == '__main__':
    main('datasets/dataset4.txt', 7)  # 选择数据集和划分数，运行c_means
