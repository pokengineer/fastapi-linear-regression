import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from pydantic import BaseModel

# Model
class my_model():
    def __init__(self):
        self.model = None

    def train_model(self ):
        # the data was generated as Y = 4 + 7 * X + noise
        read_data = np.loadtxt('linear_data.csv', delimiter=',', skiprows=1)
        X = read_data[:, 0].reshape(-1, 1) 
        y = read_data[:, 1]

        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X_new)
    
# FastAPI
app = FastAPI(title="Linear Regression API")

model = my_model()
model.train_model()

class PredictRequest(BaseModel):
    inputs: list[float]

class PredictResponse(BaseModel):
    predictions: list[float]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X_new = np.array(request.inputs).reshape(-1, 1)
    predictions = model.predict(X_new).tolist()
    return PredictResponse(predictions=predictions)

#if __name__ == "__main__":
#    user_input = input("Enter a number: ")
#    model = my_model()
#    model.train_model()
#    prediction = model.predict([[float(user_input)]]) 
#    print(round(prediction[0],2))