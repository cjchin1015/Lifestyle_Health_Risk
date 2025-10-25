import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# 1️⃣ Load environment variables (for local run)
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If deployed on Streamlit Cloud, prefer st.secrets
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# 2️⃣ Load your trained model
# ============================================================
model = joblib.load('optimized_random_forest.pkl')

# ============================================================
# 3️⃣ LabelEncoders for categorical features
# ============================================================
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

# ============================================================
# 4️⃣ Streamlit interface
# ============================================================
st.title("🏥 Lifestyle & Health Risk Prediction App")

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

# ============================================================
# 5️⃣ Prepare input dataframe
# ============================================================
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

# Compute derived features
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

exercise_map = {'none':0,'low':1,'medium':2,'high':3}
sugar_map = {'low':0,'medium':1,'high':2}
input_data['lifestyle_score'] = exercise_map[exercise] + sleep - sugar_map[sugar_intake]

# ============================================================
# 6️⃣ Encode categorical columns
# ============================================================
input_data['exercise'] = exercise_encoder.transform(input_data['exercise'].values.ravel())
input_data['sugar_intake'] = sugar_encoder.transform(input_data['sugar_intake'].values.ravel())
input_data['smoking'] = smoking_encoder.transform(input_data['smoking'].values.ravel())
input_data['alcohol'] = alcohol_encoder.transform(input_data['alcohol'].values.ravel())
input_data['married'] = married_encoder.transform(input_data['married'].values.ravel())
input_data['profession'] = profession_encoder.transform(input_data['profession'].values.ravel())
input_data['bmi_category'] = bmi_cat_encoder.transform(input_data['bmi_category'].values.ravel())

# Ensure feature order matches model
input_data = input_data[model.feature_names_in_]

# ============================================================
# 7️⃣ Model Prediction
# ============================================================
if st.button("🔍 Predict Health Risk"):
    prediction = model.predict(input_data)

    risk_map = {
        1: ("Low Risk", "green"),
        0: ("High Risk", "red")
    }

    risk_label, color = risk_map.get(prediction[0], ("Unknown", "gray"))

    st.markdown(
        f"<h3 style='color:{color};'>Predicted Risk: {risk_label}</h3>",
        unsafe_allow_html=True
    )

# ============================================================
# 8️⃣ AI Lifestyle Summary
# ============================================================
st.subheader("🧠 Step 1: Lifestyle Summary")

# ✅ Fix TypeError by extracting single values
bmi_value = float(input_data['bmi'].iloc[0])
bmi_cat_value = bmi_category(bmi_value)
lifestyle_score_value = float(input_data['lifestyle_score'].iloc[0])

st.write(f"**BMI:** {bmi_value:.2f} ({bmi_cat_value})")
st.write(f"**Lifestyle Score:** {lifestyle_score_value:.2f} (higher is better)")

summary_prompt = f"""
You are an AI health coach. Summarize the user's lifestyle profile based on the following data:

Age: {age}
Weight: {weight} kg
Height: {height} cm
Exercise level: {exercise}
Sleep hours: {sleep}
Sugar intake: {sugar_intake}
Smoking: {smoking}
Alcohol: {alcohol}
Married: {married}
Profession: {profession}
BMI: {bmi_value:.2f} ({bmi_cat_value})
Lifestyle score: {lifestyle_score_value:.2f}
"""

if st.button("🩺 Generate Lifestyle Summary"):
    with st.spinner("Analyzing your lifestyle..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI health assistant."},
                {"role": "user", "content": summary_prompt}
            ],
        )
        ai_summary = response.choices[0].message.content
        st.write(ai_summary)

    st.info("Would you like a personalized improvement plan based on this summary?")

# ============================================================
# 9️⃣ Personalized Improvement Plan
# ============================================================
if st.button("💪 Generate Personalized Improvement Plan"):
    plan_prompt = f"""
Based on this user's lifestyle and BMI profile, generate a detailed personalized plan to improve their overall health and reduce health risks.
Use a friendly, motivational tone.

Age: {age}
BMI: {bmi_value:.2f} ({bmi_cat_value})
Exercise: {exercise}
Sleep hours: {sleep}
Sugar intake: {sugar_intake}
Smoking: {smoking}
Alcohol: {alcohol}
Lifestyle score: {lifestyle_score_value:.2f}
"""

    with st.spinner("Generating personalized plan..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional health and lifestyle advisor."},
                {"role": "user", "content": plan_prompt}
            ],
        )
        ai_plan = response.choices[0].message.content
        st.success("Here’s your personalized plan:")
        st.write(ai_plan)

# ============================================================
# 🔟 Actionable Weekly Plan
# ============================================================
if st.button("📅 Generate Weekly Action Plan"):
    goal_prompt = f"""
The user wants a specific and actionable weekly lifestyle improvement plan including exercise, diet, sleep improvement, and stress management.

Age: {age}
BMI category: {bmi_cat_value}
Exercise level: {exercise}
Sleep hours: {sleep}
Sugar intake: {sugar_intake}
Smoking: {smoking}
Alcohol: {alcohol}
Profession: {profession}
Lifestyle score: {lifestyle_score_value:.2f}
"""
    with st.spinner("Creating your weekly action plan..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a certified lifestyle and fitness AI coach."},
                {"role": "user", "content": goal_prompt}
            ],
        )
        action_plan = response.choices[0].message.content
        st.success("Here’s your weekly action plan:")
        st.write(action_plan)
