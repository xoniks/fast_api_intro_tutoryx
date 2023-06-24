from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

app = FastAPI()


def load_model(file_name):
    loaded_model = pickle.load(open(file_name, 'rb'))
    return loaded_model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

@app.get("/home/home")
def read_root():
    return {"message": "Welcome to the ML car app API!"}

@app.post("/qelloja")
async def predict_car_value(request):
    model_file = 'egezon_car_ml.pkl'
    model = load_model(model_file)

    x_values = request.split(',')
    x_values = [int(i) for i in x_values]
    print(x_values)
    X_test = [x_values]
    y_predict = predict(model, X_test)
    y_predict = float(y_predict)
    
    return {"x_values": X_test, 'predicted': y_predict}
