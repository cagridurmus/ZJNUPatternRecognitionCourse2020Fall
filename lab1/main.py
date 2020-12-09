from numpy import *
import matplotlib.pyplot as plt


def load_file(filename, spliter='\t'):
    data = []
    with open(filename) as fr:
        for line in fr.readlines():
            line = line.strip().split(spliter)
            line = list(map(float, line))
            data.append(line)
    return data


def dist(a, b):
    return sqrt(sum(power(a - b, 2)))


def rand_cent(data, k):
    n = shape(data)[1]
    ct = mat(zeros((k, n)))
    for j in range(n):
        min_j, max_j = min(data[:, j]), max(data[:, j])
        delta = float(max_j - min_j)
        ct[:, j] = min_j + delta * random.rand(k, 1)
    return ct


def k_means(dt, k):
    m = shape(dt)[0]
    ca = mat(zeros((m, 1)))
    ct = rand_cent(dt, k)
    flag = True

    while flag:
        flag = False
        for j in range(m):
            min_i, min_d = 0, dist(ct[0, :], dt[j, :])

            for i in range(1, k):
                d = dist(ct[i, :], dt[j, :])
                if d < min_d:
                    min_i, min_d = i, d

            if ca[j, 0] != min_i:
                ca[j, 0] = min_i
                flag = True

        for cent in range(k):
            p = dt[nonzero(ca == cent)[0]]
            ct[cent, :] = mean(p, axis=0)

    return ct, ca


def make_plot(data, ct, ca):
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


def main():
    dataset = mat(load_file('image-tryouts/dataset4.txt', spliter=','))
    centroid, assignment = k_means(dataset, 7)
    print(centroid)
    make_plot(dataset, centroid, assignment)
    # plt.savefig('fix.jpg', dpi=550)
    plt.show()


if __name__ == '__main__':
    main()
