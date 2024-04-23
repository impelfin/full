#!/usr/bin/env python

def min(a, b):
        return b if a > b else a

a = input("Input first number: ")
b = input("Input second number: ")

print("{} vs {} : Min number = {}".format(a, b, min(a, b)))
