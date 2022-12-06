import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

league = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster']

st.write('''
# Predict league rank of a player in Star-Craft
''')

st.write(league)

st.sidebar.header("Parameters")

def user_input():
    NumberOfPACs = st.sidebar.slider('Number of PACs (unit : 10-3)',0.0,8.0,2.0)/1000
    ActionsInPAC = st.sidebar.slider('Actions in PACs',2,19,8)
    GapBetweenPACs = st.sidebar.slider('Gap between PACs',5,240,133)
    Apm = st.sidebar.slider('APM',20,400,156)
    ActionLatency = st.sidebar.slider('Action Latency',20,180,76)
    TotalMapExplored = st.sidebar.slider('Total Map Explored',5,60,30)
    UniqueHotkeys = st.sidebar.slider('Unique Hot keys',0,10,2)

    data={'NumberOfPACs':NumberOfPACs,
    'ActionsInPAC':ActionsInPAC,
    'APM':Apm,
    'UniqueHotkeys':UniqueHotkeys,
    'GapBetweenPACs':GapBetweenPACs,
    'ActionLatency':ActionLatency,
    'TotalMapExplored':TotalMapExplored,
    }

    player_features = pd.DataFrame(data,index=[0])
    return player_features

def calcul_res(val):
    res = val//10 + 1 if val%10 > 5 else 0

    if res < 1:
        return league[0]
    elif res > 7:
        return league[6]

    return league[int(res - 1)]
    

df = user_input()

st.write(df)

# load the model from disk
loaded_model = pickle.load(open("model_lassoCV.sav", 'rb'))

result = loaded_model.predict(df)

st.write("The league of player (esti. 40%): " + str(calcul_res(result)))

st.write("PS : please make a few change to make the label of the league change...")
st.write(" ")
st.write('We use lasso model for classification using the dataset SkillCraft..')