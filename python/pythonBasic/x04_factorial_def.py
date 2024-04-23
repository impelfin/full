#!/usr/bin/env python

def factorial(x):
    if x == 0:
        return 1
    else:
        return x *  factorial(x - 1)

input = int(input("Input number : "))

print(f'{input} factorial = {factorial(input)}')