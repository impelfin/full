#!/usr/bin/env python

wordinfo = {'세탁기' : 50, '선풍기' : 30, '청소기' : 40, '냉장고' : 60, }

myxticks = sorted(wordinfo, key=wordinfo.get, reverse=True)
print(myxticks)

revers_key = sorted(wordinfo.keys(), reverse=True)
print(revers_key)

chardata = sorted(wordinfo.values(), reverse=True)
print(chardata)
