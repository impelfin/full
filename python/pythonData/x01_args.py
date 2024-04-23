#!/usr/bin/env python

def fullName(*Names):
    for name in Names:
        print("%s %s" % (name[0], name[1:3]), end=" ")
    print("\n")

fullName('홍길동')
fullName('홍길동','이길동')
fullName('홍길동','이길동','김길동')

