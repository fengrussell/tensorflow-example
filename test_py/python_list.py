# -*- coding: utf-8 -*-
import time
from datetime import datetime


def list_test():
    list_num = [1, 2, 3]
    print ", ".join(map(lambda x: '%d' % x, list_num))


def list_bool_test():
    list1 = []
    if list1:
        print("list isn‘t empty")
    else:
        print("list is empty")


def list_enum_test():
    list1 = [5, 4, 3, 2, 1]
    for (i, e) in enumerate(list1):
        print (i, e)
    list2 = [e for (i, e) in enumerate(list1) if False]
    for l in list2:
        print l


def get_tuple(flag):
    if flag:
        return None

    return 1, 2


if __name__ == "__main__":
    # list_test()
    # list_bool_test()
    list_enum_test()
    # print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

