# -*- coding: utf-8 -*-

import contextlib

@contextlib.contextmanager
def context1():
    print("context1 begin ....")
    # python低版本不支持return
    yield
    print("context1 end ....")


def process():
    print("process running ...")


if __name__ == "__main__":
    with context1() as c1:
        process()
