all : appdep getdep getsvc getpod
restart : delsvc deldep appdep getdep getsvc getpod
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
	kubectl delete svc mario 
deldep:
	kubectl delete deployment mario 
