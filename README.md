# Knative service for the inference service of the msv2 app

aws lambda like serverless solution for deploying a onCall function in k3s cluster, cold start and grace period of an hours, running on the gpu node.

```sh
kubectl port-forward -n glasgow-prod svc/minio-service 9000:9000

uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

```sh
docker build -t hatchikin/msv2-inference:v0.16 .
```

