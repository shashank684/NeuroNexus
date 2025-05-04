import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout='centered')

# Background and styling
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://wallpapercave.com/wp/0swzmR9.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
background: rgba(0,0,0,0.5);
}

h1, h2, h3, h4, h5, h6, p, label, .stButton button {
color: #f0f0f0;
text-shadow: 1px 1px 2px #000;
}

.stButton button {
background-color: #FF8C00;
color: white;
border-radius: 10px;
padding: 0.7em 2em;
transition: 0.3s;
font-size: 1.2rem;
}

.stButton button:hover {
background-color: #FFA500;
transform: scale(1.05);
}

input, select, textarea {
background-color: #ffffffdd;
color: #000;
border-radius: 8px;
}

.stNumberInput input {
background-color: #ffffffdd;
color: #000;
}

.stSelectbox div {
color: #000;
}

.stCaption {
color: #e0e0e0;
}

@keyframes fadeIn {
  0% { opacity: 0; transform: translateY(-20px);}
  100% { opacity: 1; transform: translateY(0);}
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model/titanic_model.pkl')

model = load_model()

# Initialize session state for navigation and button clicks
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'start_prediction_clicked' not in st.session_state:
    st.session_state.start_prediction_clicked = False
if 'predict_survival_clicked' not in st.session_state:
    st.session_state.predict_survival_clicked = False

# Page 1: Home
if st.session_state.page == 'home':
    st.markdown("<h1 style='text-align: center; font-size: 70px; color: white; animation: fadeIn 2s;'>🚢 Titanic Survival Prediction</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

    if st.button("🌊 Start Prediction"):
        st.session_state.start_prediction_clicked = True
        st.session_state.page = 'predict'
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Page 2: Prediction
elif st.session_state.page == 'predict':

    st.markdown("<h1 style='text-align: center; color: white; animation: fadeIn 1.5s;'>📝 Passenger Details</h1>", unsafe_allow_html=True)

    # User input form
    def get_user_input():
        pclass = st.selectbox("🚢 Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3], index=2)
        sex = st.selectbox("⚥ Sex", ["male", "female"])
        age = st.number_input("🎂 Age (years)", 0.42, 90.0, 30.0)
        sibsp = st.number_input("👨‍👩‍👧‍👦 Number of siblings/spouses aboard", 0, 8, 0)
        parch = st.number_input("👵 Number of parents/children aboard", 0, 6, 0)
        fare = st.number_input("💷 Ticket (£)", 0.0, 600000000.0, 32.2)
        embarked = st.selectbox("⚓ Port of Embarkation", ["S", "C", "Q"], index=0)
        return {'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}

    def predict_survival(user_input):
        df = pd.DataFrame([user_input])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return prediction, probability

    user_input = get_user_input()

    if st.button("🔍 Predict Survival"):
        survival, prob = predict_survival(user_input)
        st.write("----")
        if survival == 1:
            st.markdown(f"<h2 style='color:#00FF7F; text-align:center; font-size: 36px;'>🌟 Passenger would SURVIVE!<br>Probability: {prob:.2%}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#FF6347; text-align:center; font-size: 36px;'>💀 Passenger would NOT survive.<br>Probability: {prob:.2%}</h2>", unsafe_allow_html=True)

    st.write("")
    if st.button("⬅ Back to Home"):
        st.session_state.page = 'home'
        st.session_state.start_prediction_clicked = False
        st.session_state.predict_survival_clicked = False
        st.rerun()
