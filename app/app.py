import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load your trained model
import joblib
model = joblib.load('optimized_random_forest.pkl')


# 2️⃣ Define LabelEncoders for categorical features (must match training)
exercise_encoder = LabelEncoder()
exercise_encoder.classes_ = ['high','low','medium','none']

sugar_encoder = LabelEncoder()
sugar_encoder.classes_ = ['high','low','medium']

smoking_encoder = LabelEncoder()
smoking_encoder.classes_ = ['no','yes']

alcohol_encoder = LabelEncoder()
alcohol_encoder.classes_ = ['no','yes']

married_encoder = LabelEncoder()
married_encoder.classes_ = ['no','yes']

profession_encoder = LabelEncoder()
profession_encoder.classes_ = ['artist','driver','engineer','farmer','office_worker','teacher']

bmi_cat_encoder = LabelEncoder()
bmi_cat_encoder.classes_ = ['normal','obese','overweight','underweight']

# 3️⃣ Streamlit input form for raw features
st.title("Lifestyle & Health Risk Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
exercise = st.selectbox("Exercise level", ['none','low','medium','high'])
sleep = st.number_input("Sleep hours", min_value=0, max_value=24, value=7)
sugar_intake = st.selectbox("Sugar intake", ['low','medium','high'])
smoking = st.selectbox("Smoking", ['no','yes'])
alcohol = st.selectbox("Alcohol", ['no','yes'])
married = st.selectbox("Married", ['no','yes'])
profession = st.selectbox("Profession", ['office_worker', 'teacher', 'artist', 'farmer', 'driver', 'engineer'])

# 4️⃣ Convert inputs into a DataFrame
input_data = pd.DataFrame({
    'age':[age],
    'weight':[weight],
    'height':[height],
    'exercise':[exercise],
    'sleep':[sleep],
    'sugar_intake':[sugar_intake],
    'smoking':[smoking],
    'alcohol':[alcohol],
    'married':[married],
    'profession':[profession]
})

# 5️⃣ Compute derived features
input_data['height_m'] = input_data['height'] / 100
input_data['bmi'] = input_data['weight'] / (input_data['height_m'] ** 2)

def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

input_data['bmi_category'] = input_data['bmi'].apply(bmi_category)
input_data['weight_height_ratio'] = input_data['weight'] / input_data['height']

# Example lifestyle score (adjust as per your training logic)
exercise_map = {'none':0,'low':1,'medium':2,'high':3}
sugar_map = {'low':0,'medium':1,'high':2}
input_data['lifestyle_score'] = exercise_map[exercise] + sleep - sugar_map[sugar_intake]

# 6️⃣ Encode categorical features safely
# Make sure to fit LabelEncoders in the correct order
exercise_encoder = LabelEncoder()
exercise_encoder.fit(['none','low','medium','high'])

sugar_encoder = LabelEncoder()
sugar_encoder.fit(['low','medium','high'])

smoking_encoder = LabelEncoder()
smoking_encoder.fit(['no','yes'])

alcohol_encoder = LabelEncoder()
alcohol_encoder.fit(['no','yes'])

married_encoder = LabelEncoder()
married_encoder.fit(['no','yes'])

profession_encoder = LabelEncoder()
profession_encoder.fit(['office_worker','teacher','artist','farmer','driver','engineer'])

bmi_cat_encoder = LabelEncoder()
bmi_cat_encoder.fit(['underweight','normal','overweight','obese'])

# Transform columns safely using .values.ravel()
input_data['exercise'] = exercise_encoder.transform(input_data['exercise'].values.ravel())
input_data['sugar_intake'] = sugar_encoder.transform(input_data['sugar_intake'].values.ravel())
input_data['smoking'] = smoking_encoder.transform(input_data['smoking'].values.ravel())
input_data['alcohol'] = alcohol_encoder.transform(input_data['alcohol'].values.ravel())
input_data['married'] = married_encoder.transform(input_data['married'].values.ravel())
input_data['profession'] = profession_encoder.transform(input_data['profession'].values.ravel())
input_data['bmi_category'] = bmi_cat_encoder.transform(input_data['bmi_category'].values.ravel())


# 7️⃣ Ensure feature order matches the model
input_data = input_data[model.feature_names_in_]

# 8️⃣ Make prediction with colored output
if st.button("Predict"):
    prediction = model.predict(input_data)
    
    # Map numeric prediction to human-readable label and color
    risk_map = {
        1: ("Low Risk", "green"),
        0: ("High Risk", "red")
    }
    
    risk_label, color = risk_map.get(prediction[0], ("Unknown", "gray"))
    
    # Display as a colored badge
    st.markdown(
        f"<h3 style='color:{color};'>Predicted Risk: {risk_label}</h3>",
        unsafe_allow_html=True
    )


import streamlit as st

st.markdown("## 🧠 AI-Powered Personalized Health Plan")

# Ask user for input
user_input = st.text_area(
    "Describe your lifestyle, health goals, or any specific concerns:",
    placeholder="Example: I exercise twice a week, sleep 6 hours, eat out often, and want to lose weight..."
)

if st.button("Generate My Personalized Plan"):
    if user_input.strip() == "":
        st.warning("Please describe your lifestyle or goals first.")
    else:
        with st.spinner("Generating personalized plan..."):
            try:
                # Call OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a certified health coach and nutrition advisor."},
                        {"role": "user", "content": f"Create a personalized health improvement plan for: {user_input}. Include fitness, nutrition, sleep, and stress advice."}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )

                ai_reply = response["choices"][0]["message"]["content"]
                st.success("✅ Your Personalized Health Plan:")
                st.write(ai_reply)

            except Exception as e:
                st.error(f"Error: {e}")






