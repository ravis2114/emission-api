from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from model_arch import Net
from utils import input_template

#loading model
model = Net(46,4)
model.load_state_dict(torch.load('model.pth'))
model.eval()

#load dictionary from json file
with open('start_port_dict.json', 'r') as fp:
    start_port_dict = json.load(fp)
    start_port_dict = {float(k):v for k,v in start_port_dict.items()}
with open('end_port_dict.json', 'r') as fp:
    end_port_dict = json.load(fp)
    end_port_dict = {float(k):v for k,v in end_port_dict.items()}

#load scaler
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)
with open('scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)


class InputData(BaseModel):
    Start_Port_Id : float
    End_Port_Id : float
    Total_Distance: float
    Length: float
    Breadth : float
    Depth : float
    Draft : float
    Grt : float
    Dwt : float
    Total_Cargo_Onboard : float
    Cargo_Mt : float
    Avg_Actual_Speed_Logged : float
    Avg_Draft_Fore : float
    Avg_Draft_After : float
    Avg_Displacement : float
    Tot_ME_Run_Hours : float
    Log_Time_Duration : float
    Avg_AE_Power : float
    Vessel_Type : str
    Sub_Type : str
def api_input_transform(input_template, api_input):
    input_template = input_template.copy()
    input_template[' Start_Port_Id'] = api_input['Start_Port_Id']
    input_template[' End_Port_Id'] = api_input['End_Port_Id']
    input_template[' Total_Distance'] = api_input['Total_Distance']
    input_template['Length'] = api_input['Length']
    input_template['Breadth'] = api_input['Breadth']
    input_template['Depth'] = api_input['Depth']
    input_template['Draft'] = api_input['Draft']
    input_template['Grt'] = api_input['Grt']
    input_template['Dwt'] = api_input['Dwt']
    input_template[' Total_Cargo_Onboard'] = api_input['Total_Cargo_Onboard']
    input_template[' Cargo_Mt'] = api_input['Cargo_Mt']
    input_template[' Avg_Actual_Speed_Logged'] = api_input['Avg_Actual_Speed_Logged']
    input_template[' Avg_Draft_Fore'] = api_input['Avg_Draft_Fore']
    input_template[' Avg_Draft_After'] = api_input['Avg_Draft_After']
    input_template['Avg_Displacement'] = api_input['Avg_Displacement']
    input_template[' Tot_ME_Run_Hours'] = api_input['Tot_ME_Run_Hours']
    input_template['Log_Time_Duration'] = api_input['Log_Time_Duration']
    input_template['Avg_AE_Power'] = api_input['Avg_AE_Power']
    input_template['Vessel_Type_' + api_input['Vessel_Type']] = 1
    input_template['Sub_Type_' + api_input['Sub_Type']] = 1
    return input_template
def preprocess_api_input(final_api_input):
    final_api_input = final_api_input.copy()
    final_api_input[' Start_Port_Id'] = final_api_input[' Start_Port_Id'].map(start_port_dict)
    final_api_input[' End_Port_Id'] = final_api_input[' End_Port_Id'].map(end_port_dict)
    print(final_api_input)
    final_api_input = scaler.transform(final_api_input)
    final_api_input = torch.from_numpy(final_api_input.astype(np.float32))
    return final_api_input

app = FastAPI()

# CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])



@app.post("/api/v1/predict")
async def predict(api_input: InputData):
    print(api_input)
    api_input = api_input.dict()
    final_api_input = api_input_transform(input_template, api_input)
    final_api_input = pd.DataFrame(final_api_input, index=[0])
    print(final_api_input)
    final_api_input = preprocess_api_input(final_api_input)
    print(final_api_input)
    with torch.no_grad():
        preds = model(final_api_input)
        preds = scaler_y.inverse_transform(preds)
        pred_df = pd.DataFrame(preds, columns=['Total_Fuel_Consumption', 'Total Co2', 'Total Nox', 'Total Sox'])
        pred_df = pred_df.round(2)
        pred_df = pred_df.to_dict('records')
    return pred_df
