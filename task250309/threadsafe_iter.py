#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
'''
这是一个线程安全的迭代器装饰器。

当多个线程需要同时从同一个生成器获取数据时，可能会出现数据竞争的情况，导致不可预测的结果。
使用这个装饰器可以确保在多线程环境下，生成器能够安全地被多个线程使用。
'''

class threadsafe_iter:
    '''
    threadsafe_iter 类：

    这个类包装了一个迭代器对象 it，并添加了一个锁 lock。
    __iter__ 方法返回对象自身，使其成为一个迭代器。
    __next__ 方法使用 with self.lock 获取锁，保证在同一时刻只有一个线程可以访问 it.__next__()，从而实现线程安全的迭代。

    '''
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    '''
    threadsafe_generator 函数：

    这是一个装饰器函数，它接受一个生成器函数 f 作为参数。
    它定义了一个新的生成器函数 g，g 内部调用原始生成器函数 f，并使用 threadsafe_iter 包装其返回值，
    确保生成的迭代器是线程安全的。
    最后，它返回新的生成器函数 g。
    '''
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g
