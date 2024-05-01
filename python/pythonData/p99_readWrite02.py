#!/usr/bin/env python

myfile01 = open('sample02.txt', 'rt', encoding='UTF-8')
linelists = myfile01.readlines()
myfile01.close()

myfile02 = open('result.txt', 'wt', encoding='UTF-8')

total = 0
for one in linelists:
    mylist = one.split(',')
    if (int)(mylist[1]) >= 19:
        adult = '성인'
    else:
        adult = '미성년'
    text = mylist[0] + '/' + mylist[1].strip() + '/'+adult
    myfile02.write(text + '\n')
myfile02.close()

myfile03 = open('result.txt', 'rt', encoding='UTF-8')
line = 1
while line:
    line = myfile03.readline()
    print(line)
myfile03.close()
