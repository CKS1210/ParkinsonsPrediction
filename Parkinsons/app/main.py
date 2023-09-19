import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

def get_clean_data():
    data = pd.read_csv("data/parkinsons_data.csv")
    data = data.drop(['name'], axis=1)
    return data

def add_sidebar():
    st.sidebar.title("Matrix Columns")

    data = get_clean_data()

    slider_labels=[
        ("MDVP Fo(Hz)","MDVP:Fo(Hz)"),
        ("MDVP  Fhi(Hz)","MDVP:Fhi(Hz)"),
        ("MDVP: Flo(Hz)","MDVP:Flo(Hz)"),
        ("MDVP: Jitter( %)","MDVP:Jitter(%)"),
        ("MDVP: Jitter(Abs)","MDVP:Jitter(Abs)"),
        ("MDVP: RAP","MDVP:RAP"),
        ("MDVP: PPQ","MDVP:PPQ"),
        ("Jitter: DDP","Jitter:DDP"),
        ("MDVP: Shimmer","MDVP:Shimmer"),
        ("MDVP: Shimmer(dB)","MDVP:Shimmer(dB)"),
        ("Shimmer: APQ3","Shimmer:APQ3"),
        ("Shimmer: APQ5","Shimmer:APQ5"),
        ("MDVP: APQ","MDVP:APQ"),
        ("Shimmer: DDA","Shimmer:DDA"),
        ("NHR","NHR"),
        ("HNR","HNR"),
        ("RPDE","RPDE"),
        ("DFA","DFA"),
        ("spread1","spread1"),
        ("spread2","spread2"),
        ("D2","D2"),
        ("PPE","PPE")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label = label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean())
        )
    return input_dict

def get_radar_data(input_data):

    input_data = get_scaled_values(input_data)
    #st.write(input_data)

    df = pd.DataFrame(dict(
        r=[input_data['MDVP:Fo(Hz)'],input_data['MDVP:Fhi(Hz)'],input_data['MDVP:Flo(Hz)'],input_data['MDVP:Jitter(%)'],input_data['MDVP:Jitter(Abs)'],input_data['MDVP:RAP'],
           input_data['MDVP:PPQ'],input_data['Jitter:DDP'],input_data['MDVP:Shimmer'],input_data['MDVP:Shimmer(dB)'],input_data['Shimmer:APQ3'],input_data['Shimmer:APQ5'],
           input_data['MDVP:APQ'], input_data['Shimmer:DDA'], input_data['NHR'], input_data['HNR'],input_data['RPDE'], input_data['DFA'],input_data['spread1'], input_data['spread2'],
           input_data['D2'], input_data['PPE']],

        theta=["MDVP: Fo(Hz)","MDVP: Fhi(Hz)","MDVP: Flo(Hz)","MDVP: Jitter( %)","MDVP: Jitter(Abs)","MDVP: RAP","MDVP: PPQ",
               "Jitter: DDP","MDVP: Shimmer","MDVP: Shimmer(dB)","Shimmer: APQ3","Shimmer: APQ5","MDVP: APQ","Shimmer: DDA",
               "NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]))

   #,"spread1","spread2","D2","PPE"
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(width=800, height=600)

    return fig

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['status'],axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        min_val = X[key].min()
        max_val = X[key].max()
        scaled_value = (value - min_val)/(max_val-min_val)
        scaled_dict[key] = abs(scaled_value)

    return scaled_dict

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl","rb"))
    scaler = pickle.load(open("model/scaler.pkl","rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.write("<span class = 'diagnosis parkinsons'>Parkinsons</span>", unsafe_allow_html=True)
    else:
        st.write("<span class = 'diagnosis healthy'>Healthy</span>", unsafe_allow_html=True)

    st.write("Probability of getting Parkinson's Disease: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("Probability of NOT getting Parkinson's Disease: ", model.predict_proba(input_array_scaled)[0][0])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a subtitute for a professional diagnosis")

def main():
    st.set_page_config(
        page_title= "Parkinson's Disease Predictor",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded")

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    #st.write(input_data)

    with st.container():
        st.title("Parkinson's Disease Predictor")
        st.write("This app predicts using a machine learning model whether a patient has Parkinson's disease based on the measurements it receives from diagnostic tests. "
                 "You can update the measurements by hand using the sliders in the sidebar")

        col1, col2 = st.columns([4,1])
        with col1:
            radar_chart = get_radar_data(input_data)
            st.plotly_chart(radar_chart)
        with col2:
            add_predictions(input_data)


if __name__ == '__main__':
     main()