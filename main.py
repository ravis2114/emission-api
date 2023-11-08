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

# loading model
model = Net(40, 5)
model.load_state_dict(torch.load(
    'model.pth', map_location=torch.device('cpu')))
model.eval()

# load dictionary from json file
with open('start_port_dict.json', 'r') as fp:
    start_port_dict = json.load(fp)
    start_port_dict = {float(k): v for k, v in start_port_dict.items()}
with open('end_port_dict.json', 'r') as fp:
    end_port_dict = json.load(fp)
    end_port_dict = {float(k): v for k, v in end_port_dict.items()}
with open('port_name_id_dict.json', 'r') as fp:
    port_name_id_dict = json.load(fp)

# load scaler
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)
with open('scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)


class InputData(BaseModel):
    Start_Port_Id: str
    End_Port_Id: str
    Total_Distance: float
    Dwt: float
    Avg_Actual_Speed_Logged: float
    Vessel_Type: str
    Sub_Type: str
    average_ME_mcr: float
    Calm: float
    Smooth: float
    Slight: float
    Moderate: float
    Rough: float
    Very_rough: float
    High: float
    Very_high: float
    Phenomenal: float


def api_input_transform(input_template, api_input):
    input_template = input_template.copy()
    input_template[' Start_Port_Id'] = float(
        port_name_id_dict[api_input['Start_Port_Id']])
    input_template[' End_Port_Id'] = float(
        port_name_id_dict[api_input['End_Port_Id']])
    input_template[' Total_Distance'] = api_input['Total_Distance']
    input_template['Dwt'] = api_input['Dwt']
    input_template[' Avg_Actual_Speed_Logged'] = api_input['Avg_Actual_Speed_Logged']
    input_template['Vessel_Type_' + api_input['Vessel_Type']] = 1
    input_template['Sub_Type_' + api_input['Sub_Type']] = 1
    input_template['average_ME_mcr'] = api_input['average_ME_mcr']
    input_template['Calm'] = api_input['Calm']
    input_template['Smooth'] = api_input['Smooth']
    input_template['Slight'] = api_input['Slight']
    input_template['Moderate'] = api_input['Moderate']
    input_template['Rough'] = api_input['Rough']
    input_template['Very rough'] = api_input['Very_rough']
    input_template['High'] = api_input['High']
    input_template['Very high'] = api_input['Very_high']
    input_template['Phenomenal'] = api_input['Phenomenal']
    return input_template


def preprocess_api_input(final_api_input):
    final_api_input = final_api_input.copy()
    final_api_input[' Start_Port_Id'] = final_api_input[' Start_Port_Id'].map(
        start_port_dict)
    final_api_input[' End_Port_Id'] = final_api_input[' End_Port_Id'].map(
        end_port_dict)
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
        pred_df = pd.DataFrame(preds, columns=[
                               'Total_Fuel_Consumption', 'Total Co2', 'Total Nox', 'Total Sox', 'ETA'])
        pred_df = pred_df.round(2)
        pred_df = pred_df.to_dict('records')
    return pred_df


@app.get("/api/v1/port_name_id_dict")
async def port_name_id_dict_get():
    return {'port_names': list(port_name_id_dict.keys())}


@app.get("/api/v1/vessel_type")
async def vessel_type_get():
    return {'vessel_type': ['Chemical/Oil Tanker',
                            'Container Ship',
                            'General Cargo Ship',
                            'LNG Carrier',
                            'LPG Carrier',
                            'Oil Tanker',
                            'Ro-Ro Ship',
                            'Specialised Ship']}


@app.get("/api/v1/sub_type")
async def sub_type_get():
    return {'sub_type': ['Bulk Carrier',
                         'Chemical/Oil Products Tanker IMO-1',
                         'Chemical/Oil Products Tanker IMO-2',
                         'Chemical/Oil Products Tanker IMO-3',
                         'Container Ship (Fully Cellular)',
                         'Crude Tanker',
                         'Crude/Oil Products Tanker',
                         'General Cargo Self-unloader Ship',
                         'LNG DF Diesel (XDF/MEGI)',
                         'LNG Diesel',
                         'LPG Ethylene Carrier',
                         'LPG Fully Refrigerated',
                         'LPG Semi Pressurized',
                         'Ore Carrier',
                         'Ore-Bulk-Oil Carrier',
                         'Products Tanker',
                         'Ro-Ro Cargo Ship']}
