#!/usr/bin/env python

import random

class Prime(object):
    def __init__(self, num):
        self.num = num

    def isprime(self):
        for k in range(2, self.num + 1):
            if self.num % k == 0:
                break
        if k == self.num:
            return True
        else:
            return False

prime = Prime(random.randint(2, 10))
print(f'{prime.num} is Prime numbers') if prime.isprime() else print(f'{prime.num} is not Prime numbers')
