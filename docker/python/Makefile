build:
	docker build -t python .
run:
	docker run -it -d -p 8000:8000 -v /Users/lune/Documents/Github/full/docker/python/data:/root --name python python
exec:
	docker exec -it python /bin/bash
start:
	docker start python
stop:
	docker stop	python
logs:
	docker logs python
ps:
	docker ps -a
img:	
	docker images
rm:
	docker rm -f $$(docker ps -aq)
rmi:
	docker rmi -f $$(docker images -q)
