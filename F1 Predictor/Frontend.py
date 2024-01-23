import streamlit as st
import datetime as dt
import joblib
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
icon = Image.open('Logo.png')



# Page config
st.set_page_config(page_title="F1 Predictor",
                     page_icon=icon      
                   )
st.image('Logo.png')
st.title("F1 Race Result Prediction App")
st.write("""

This app predicts if your favourite driver will finish on podium/in points/or no points based on qualifying position

""")
image_path = "C://Users//keane//OneDrive//Desktop//College//3rd year//Machine Learning//Project//Logo.png"



circuits={'Albert Park Street Circuit': 0, 'Dino & Enzo Ferrari Autodrome': 1, 'Autodromo Internazionale del Mugello': 2, 'Monza National Autodrome': 3, 'Autdoromo Hermanos Rodríguez': 4, 'Autódromo Internacional do Algarve': 5, 'José Carlos Pace Autodrome': 6, 'Sakhir International Circuit': 7, 'Baku City Circuit': 8, 'Buddh International Circuit': 9, 'Gilles Villenueve Circuit': 10, 'Zandvoort Circuit': 11, 'Circuit Paul Ricard': 12, 'Barcelona-Catalunya Circuit': 13, 'Monaco Street Circuit': 14, 'Circuit de Nevers Magny-Cours': 15, 'Spa-Francorchamps Circuit': 16, 'Circuit of the Americas': 17, 'Fuji Speedway': 18, 'Hockenheimring': 19, 'Hungaroring': 20, 'Indianapolis Motor Speedway': 21, 'Istanbul Park': 22, 'Jeddah Corniche Circuit': 23, 'Korean International Circuit': 24, 'Losail International Circuit': 25, 'Marina Bay Street Circuit': 26, 'Miami International Autodrome': 27, 'Nürburgring': 28, 'Red Bull Ring': 29, 'Sepang International Circuit': 30, 'Shanghai International Circuit': 31, 'Silverstone Circuit': 32, 'Sochi Autodrom': 33, 'Suzuka International Racing Course': 34, 'Valencia Street Circuit': 35, 'Yas Marina Circuit': 36,'Las Vegas Strip Street Circuit':37}
drivers={'Alexander Albon': 0, 'Carlos Sainz': 1, 'Charles Leclerc': 2, 'Daniel Ricciardo': 3, 'Esteban Ocon': 4, 'Fernando Alonso': 5, 'George Russell': 6, 'Kevin Magnussen': 7, 'Lance Stroll': 8, 'Lando Norris': 9, 'Lewis Hamilton': 10, 'Logan Sargeant': 11, 'Max Verstappen': 12, 'Oscar Piastri': 13, 'Pierre Gasly': 14, 'Sergio Pérez': 15, 'Valtteri Bottas': 16, 'Yuki Tsunoda': 17}
constructors={'Alfa Romeo': 0, 'AlphaTauri': 1, 'Alpine F1 Team': 2, 'Ferrari': 3, 'Haas F1 Team': 4, 'McLaren': 5, 'Mercedes': 6, 'Aston Martin': 7, 'Red Bull': 8, 'Williams': 9}
drivers_dob={'Alexander Albon': dt.date(1996,3,23), 'Carlos Sainz': dt.date(1994,9,1), 'Charles Leclerc': dt.date(1997,10,16), 'Daniel Ricciardo': dt.date(1989,7,1), 'Esteban Ocon': dt.date(1996,9,17), 'Fernando Alonso': dt.date(1981,7,29), 'George Russell': dt.date(1998,2,15), 'Kevin Magnussen': dt.date(1992,10,5), 'Lance Stroll': dt.date(1998,10,29), 'Lando Norris': dt.date(1999,11,13), 'Lewis Hamilton': dt.date(1985,1,7), 'Logan Sargeant': dt.date(2000,12,31), 'Max Verstappen': dt.date(1997,9,30), 'Oscar Piastri': dt.date(2001,4,6), 'Pierre Gasly': dt.date(1996,2,7), 'Sergio Pérez': dt.date(1990,1,26), 'Valtteri Bottas': dt.date(1989,8,28), 'Yuki Tsunoda': dt.date(2000,5,11)}
vals=['Albert Park Street Circuit','Carlos Sainz',2,'McLaren',dt.date(2019, 7, 6)]
driver_confidence={'Daniel Ricciardo': 0.9575757575757576, 'Kevin Magnussen': 0.9104477611940298, 'Carlos Sainz': 0.9022988505747126, 'Valtteri Bottas': 0.9485714285714286, 'Lance Stroll': 0.8955223880597015, 'Alexander Albon': 0.9285714285714286, 'George Russell': 0.9361702127659576, 'Logan Sargeant': 0.8333333333333334, 'Fernando Alonso': 0.9323308270676692, 'Lando Norris': 0.9574468085106383, 'Oscar Piastri': 0.9166666666666666, 'Charles Leclerc': 0.8956521739130435, 'Esteban Ocon': 0.9426229508196722, 'Pierre Gasly': 0.9159663865546218, 'Lewis Hamilton': 0.9827586206896552, 'Max Verstappen': 0.9142857142857144, 'Sergio Pérez': 0.976878612716763, 'Yuki Tsunoda': 0.9285714285714286}
constructor_reliability={'Alpine F1 Team': 0.5273775216138328, 'Williams': 0.4257142857142857, 'McLaren': 0.5072046109510087, 'Ferrari': 0.8714285714285714, 'Aston Martin': 0.5914285714285714, 'Mercedes': 0.9428571428571428, 'AlphaTauri': 0.4755043227665706, 'Alfa Romeo': 0.3591954022988506, 'Red Bull': 0.8171428571428572, 'Haas F1 Team': 0.3729903536977492}
def RaceResult(arr):
    test=[]
    test.append(circuits[arr[0]])
    test.append(arr[2])
    test.append(constructors[arr[3]])
    test.append(drivers[arr[1]])
    test.append(driver_confidence[arr[1]])
    test.append(constructor_reliability[arr[3]])
    #Calculating the age during GP
    dob=drivers_dob[arr[1]]
    age_in_days=abs(arr[4]-dob)
    age_in_days=int(str(age_in_days).split(' ')[0])
    test.append(age_in_days)
    test=np.array(test).reshape(1, -1).tolist()
    return test
def model_run(test):
    df=pd.DataFrame(test)
    loaded_rf = joblib.load("final_rf.joblib")
    pred=loaded_rf.predict(df.values)
    if(pred[0]==1):
        return "Podium"
    elif(pred[0]==2):
        return "Points"
    else:
        return "No Points"

def user_input():
    ciruit=st.selectbox('Circuit',
                          ('Sakhir International Circuit',
                           'Jeddah Corniche Circuit',
                           'Albert Park Street Circuit',
                           'Baku City Circuit',
                           'Miami International Autodrome',
                           'Dino & Enzo Ferrari Autodrome',
                           'Monaco Street Circuit',
                           'Barcelona-Catalunya Circuit',
                           'Gilles Villenueve Circuit',
                           'Red Bull Ring',
                           'Silverstone Circuit','Hungaroring',
                           'Spa-Francorchamps Circuit',
                           'Zandvoort Circuit',
                           'Monza National Autodrome',
                           'Marina Bay Street Circuit',
                           'Suzuka International Racing Course',
                           'Losail International Circuit',
                           'Circuit of the Americas',
                           'Autdoromo Hermanos Rodríguez',
                           'José Carlos Pace Autodrome',
                           'Las Vegas Strip Street Circuit',
                           'Yas Marina Circuit'),placeholder=' ',index=None)
    driver=st.selectbox("Driver",
                        ('Daniel Ricciardo', 'Kevin Magnussen', 'Carlos Sainz',
                        'Valtteri Bottas', 'Lance Stroll', 'George Russell',
                        'Lando Norris', 'Oscar Piastri', 'Yuki Tsunoda',
                        'Charles Leclerc', 'Lewis Hamilton',
                        'Max Verstappen', 'Pierre Gasly', 'Alexander Albon',
                        'Sergio Pérez', 'Esteban Ocon',
                        'Logan Sargeant','Fernando Alonso'),placeholder=' ',index=None)
    qualifying=st.selectbox("Qualifying Position",
                            (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),placeholder=' ',index=None)
    constructor=st.selectbox("Constructor",('Mercedes',
                                            'Red Bull',
                                            'Ferrari',
                                            'McLaren',
                                            'Aston Martin',
                                            'Alpine F1 Team',
                                            'AlphaTauri',
                                            'Alfa Romeo',
                                            'Haas F1 Team',
                                            'Williams')
                                            ,placeholder=' ',index=None)
    GPdate = st.date_input("GP Date",dt.date(2023, 7, 9))
    array=[ciruit,driver,qualifying,constructor,GPdate]
    
    # All this after button click
     
    prediction=st.button("Predict",type='primary')
    if prediction:
        sample=RaceResult(array)
        value=model_run(sample)  
        st.title(f"Prediction:")
        st.write (value)

        
    
        
user_input()
