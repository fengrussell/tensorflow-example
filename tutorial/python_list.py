# -*- coding: utf-8 -*-


def list_test():
    list_num = [1, 2, 3]
    print ", ".join(map(lambda x: '%d' % x, list_num))


if __name__ == "__main__":
    list_test()
