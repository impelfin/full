#!/usr/bin/env python

import cx_Oracle
from xml.etree.ElementTree import parse

cx_Oracle.init_oracle_client(lib_dir='/usr/local/OracleXE/instantclient_19_22')

conn = None
cur = None

tree = parse('xmlEx_04_total.xml')
myroot = tree.getroot()

try:
    # id/pw@hostname:port_number/sid
    loginfo = 'hr/1234@192.168.1.84:1521/xe'
    conn = cx_Oracle.connect(loginfo, encoding='utf-8')
    cur = conn.cursor()

    items = myroot.findall('item')

    for oneitem in items:
        sql = " insert into shop"
        sql += " values('"
        sql += oneitem[0].text + "', '"
        sql += oneitem[1].text + "', '"
        sql += oneitem[2].text + "', '"
        sql += oneitem[3].text + "', '"
        sql += oneitem[4].text + "', '"
        sql += oneitem[5].text + "', '"
        sql += oneitem[6].text + "', '"
        sql += oneitem[7].text + "'"
        sql += " )"
        cur.execute(sql)
    conn.commit()

except Exception as err:
    if conn != None:
        conn.rollback()
    print(err)

finally:
    if cur != None:
        cur.close()

    if conn != None:
        conn.close()

print('finished..')