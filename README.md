# 🧠 AI Lifestyle & Health Risk Prediction App  

An AI-powered Streamlit web application that predicts an individual’s **health risk level** (High Risk / Low Risk) based on lifestyle and physiological factors such as diet, sleep quality, physical activity, BMI, smoking, and alcohol consumption.  

🔗 **Live Demo:** [https://lifestyle-health-risk-ai-final.streamlit.app/](https://lifestyle-health-risk-ai-final.streamlit.app/)  

---

## 💡 Problem Statement  
Lifestyle-related diseases such as obesity, diabetes, hypertension, and heart disease are increasing rapidly worldwide. Many people are unaware of their potential health risks due to a lack of accessible tools for early detection.  

This project aims to develop an intelligent AI solution that predicts a person’s health risk based on daily habits and body metrics, while also providing interpretable insights and personalized recommendations for better lifestyle management.

---

## 🎯 Objectives  
1. Build an AI model that classifies users into **High Risk** or **Low Risk** health categories based on lifestyle and physiological data.  
2. Identify key behaviors (e.g., diet, exercise, sleep) that most influence health risk predictions.  
3. Generate **AI-driven personalized health improvement plans** using LLM-based recommendations.  

---

## 🧩 Proposed Solution / Approach  
The project integrates **traditional AI** for predictive modeling and **generative AI** for personalized recommendations:  

- **Random Forest Classifier** for risk prediction based on structured input data.  
- **LLM (Generative AI)** integration to translate analytical results into actionable lifestyle advice.  

---

## 📊 Data  
**Source:** [Kaggle – Lifestyle and Health Risk Prediction Dataset](https://www.kaggle.com/datasets/miadul/lifestyle-and-health-risk-prediction)  
**Description:** A realistic synthetic dataset designed for predicting health risk levels based on individual lifestyle indicators.  

---

## ⚙️ Methodology  

### 1. Data Preprocessing  
- Handled missing values and categorical variables  
- Normalized numerical features for model compatibility  

### 2. Exploratory Data Analysis (EDA)  
- Analyzed correlations between BMI, sleep, smoking, and health risk  
- Visualized feature distributions and patterns  

### 3. Model Building  
- Compared multiple models
- Selected **Optimized Random Forest** for highest accuracy and interpretability  

### 4. Deployment  
- Built and deployed an interactive **Streamlit web app**  
- Allows users to input their health/lifestyle data and receive:  
  - Risk classification  
  - BMI calculation  
  - Key factor insights  

---

## 💻 Tools & Technologies  

| Category | Tools Used |
|-----------|-------------|
| **Development Environment** | Google Colab |
| **Language** | Python |
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn, joblib |
| **Generative AI** | OpenAI API / LLM Integration |
| **Frontend & Deployment** | Streamlit |

---

## 🚀 Expected Outcomes  
- Interactive Streamlit app for health risk prediction  
- Real-time classification as *High Risk* or *Low Risk*  
- BMI score and key factor visualizations  
- LLM-powered personalized lifestyle recommendations  

---

## 🔮 Potential Next Enhancement  
- Integrate a **personalized health plan generator** using advanced LLM-based recommendations.  
- Include user history tracking and progress analytics.  

---

## 📬 Contact  
For inquiries or collaboration:  
📧 **cjchin1015@gmail.com**  
