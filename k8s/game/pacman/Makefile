all : appdep getdep getsvc getpod
restart : delsvc deldep delpod appdep getdep getsvc getpod
stop: delsvc deldep delpod
getnode:
	kubectl get nodes
appdep:
	kubectl apply -f deployment.yaml
getdep:
	kubectl get deployment
getsvc:
	kubectl get svc -o wide
getpod:
	kubectl get pod -o wide
delsvc:
	kubectl delete svc pacman 
deldep:
	kubectl delete deployment pacman
delpod:
	kubectl get pod | grep pacman | awk '{print $$1}' | xargs -I {} kubectl delete pod {} --force
desdep:
	kubectl describe deployment pacman
dessvc:
	kubectl describe svc pacman
despod:
	kubectl describe pod pacman
	
