#!/usr/bin/env python

class Factorial(object):
    def __init__(self, x):
        self.x = x
    def factorial(self):
        n = 1
        for i in range(1, self.x + 1):
            n *= i
        return n

input = int(input("Input number : "))
fact = Factorial(input)
print(f'{input} factorial = {fact.factorial()}')