#!/usr/bin/env python

def nolamda(x, y):
    return 3 * x + 2 * y

x, y = 3, 5

yeslamda = lambda x, y: 3 * x + 2 * y
result = yeslamda(x, y)
print("람다 방식 : %d" % (result))

