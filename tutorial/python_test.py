# !/usr/bin/env python2
# -*- coding: utf-8 -*-


def zip_test():
    a = [1, 2, 3]
    b = [4, 5, 6]
    print zip(a, b)  # [(1, 4), (2, 5), (3, 6)]

    c = [(1, 4), (2, 5), (3, 6)]
    print zip(*c)  # [(1, 2, 3), (4, 5, 6)]

    c1 = [[1, 4], [2, 5], [3, 6]]
    print zip(*c1)  # [(1, 2, 3), (4, 5, 6)]

    d = [1, 4]
    e = [2, 5]
    f = [3, 6]
    print zip(d, e, f)  # *c/c1的拆解 [(1, 2, 3), (4, 5, 6)]


if __name__ == "__main__":
    zip_test()
