#!/usr/bin/env python

def fullName(**kwargs):
    for key, value in kwargs.items():
        if 'master' in kwargs.keys():
            print(f'Hello Moon, How are you?')
        else:
            print(f'{key} is {value}')

fullName(name="Moon")
fullName(master="Moon")