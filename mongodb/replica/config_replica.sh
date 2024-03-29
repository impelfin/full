if [ -d /replica/data ] ; then
	rm -rf /replica/data
else
	mkdir data
	mkdir -pv /replica/data/master
	mkdir -pv /replica/data/slave1
	mkdir -pv /replica/data/slave2
	mkdir -pv /replica/data/arbiter
	touch /replica/data/master/master.log
	touch /replica/data/slave1/slave1.log
	touch /replica/data/slave2/slave2.log
	touch /replica/data/arbiter/arbiter.log
fi

/replica/mongodb/bin/mongod --config /replica/master.conf &
/replica/mongodb/bin/mongod --config /replica/slave1.conf &
/replica/mongodb/bin/mongod --config /replica/slave2.conf &
/replica/mongodb/bin/mongod --config /replica/arbiter.conf &
netstat -ntlp | grep mongo
sleep 3s

/replica/mongodb/bin/mongo --port 10000 < rs_start
