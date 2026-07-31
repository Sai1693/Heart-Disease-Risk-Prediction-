import pickle
import numpy as np


model = pickle.load(
    open(
        "models/heart_disease_model.pkl",
        "rb"
    )
)


scaler = pickle.load(
    open(
        "models/scaler.pkl",
        "rb"
    )
)



def predict_heart_risk(data):

    data = np.array(data).reshape(1,-1)

    data = scaler.transform(data)


    result = model.predict(data)


    probability = model.predict_proba(data)


    return result[0], probability[0][1]
