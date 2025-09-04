# fastapi-linear-regression

**Linear Regression model** inside a Docker container using **FastAPI**

---
This project demonstrates how to containerize a machine learning model and expose it as a REST API.  
The model is a simple **linear regression with one feature** and one target, trained on a dataset generated as: <br>
y = 4 + 7x + $\epsilon$ <br>
with $\epsilon$ as noise, and $\mathbb{E}(\epsilon) = 0$

---

## Getting Started

### 1. Build and run with Docker Compose
```
docker-compose up --build
```
- Build the Docker image
- Start a container running the FastAPI app at http://localhost:8000
  
### 2. Test the API
- option 1: fastapi interactive docs
Open your browser: http://localhost:8000/docs
<img width="1424" height="829" alt="image" src="https://github.com/user-attachments/assets/b48b920d-43a1-4a68-8c4d-c96e1afd0646" />

 - option 2: curl request 
```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    7 , 9 , 4
  ]
}'
```
example response
```
{
  "predictions": [ 53.13, 67.11, 32.17 ]
}
```
<img width="1254" height="516" alt="image" src="https://github.com/user-attachments/assets/4c8175aa-c92a-47fd-b98f-2f043b6cc255" />


