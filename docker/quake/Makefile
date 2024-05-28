build:
	docker build -t treyyoder/quakejs:latest .
run:
	docker run -d --name quake -e HTTP_PORT=8080 -p 8080:80 -p 27960:27960 treyyoder/quakejs:latest
logs:
	docker logs quake
ps:
	docker ps -a
img:
	docker images
rm:
	docker rm -f $$(docker ps -aq)
rmi:
	docker rmi $$(docker images -q)
