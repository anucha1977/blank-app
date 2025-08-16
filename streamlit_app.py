import streamlit as st 
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder # Needed to inverse transform predictions
import random
from numpy.random import default_rng as rng

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

with open('style2.css') as f:
   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#st.title("Gestão do Projeto XYZ")

#######################################
# STREAMLIT LAYOUT
#######################################

Titlebar, = st.columns(1)
Line1L,Line1R = st.columns(2)
Line2L,Line2R = st.columns(2)
top_left_column, top_right_column = st.columns([0.8,0.2])
bottom_left_column, bottom_right_column = st.columns(2)

with Titlebar:
    # อินเทอร์เฟซผู้ใช้
    st.title("Diabetes Risk Predictor")
    st.write("ใส่ข้อมูลสุขภาพของคุณ เพื่อประเมินความเสี่ยงโรคเบาหวาน (DM)")

with Line1L:
    st.subheader("Home")
    st.write("Health Risk Predictor \n This application predicts health risk scores based \n on user-entered information.")

    st.progress(50, text="Progresso")
    
    column_1, column_2, column_3, column_4 = st.columns(4)

    with column_1:
        st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

    with column_2:
        st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

    with column_3:
        st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
        
    with column_4:
        st.metric(label="Encerramento", value="40")
       
    st.subheader("Home")
    df = pd.DataFrame(
        {
            "Name": ["App A", "App B"],
            "Command": ["**st.table**", "*st.dataframe*"],
            "Type": ["`static`", "`interactive`"],
            "Docs": [
                "[:rainbow[docs]](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)",
                "[:open_book:](https://docs.streamlit.io/develop/api-reference/data/st.table)",
            ]
        }
    )

    st.dataframe(
        df,
        column_config={
            "Name": "App name",
            "Command": "Github Stars",
            "Type": "App URL",
            "Docs": "Views (past 30 days)"
        },
        hide_index=True,
    )

    st.subheader("Sales Overview")
    st.write("Prediction Result Low Risk Probability: 0.15 Advice Your risk level is low, maintain a healthy ifestyle to keep it that way.")

    st.subheader("Sales Overview")
    st.write("This section provides an overview of sales performance.")

with Line1R:
    # โหลดโมเดลที่ฝึกไว้แล้ว
    @st.cache_resource # Using st.cache_resource as recommended
    def load_model():
        try:
            with open("xgboost_dm_model.pkl", "rb") as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            st.error("Model file 'xgboost_dm_model.pkl' not found. Please train the model first.")
            return None

    # โหลด LabelEncoder หรือสร้าง mapping สำหรับแปลงผลทำนายกลับ
    # จากเซลล์ 0nWeTsnKDoQs, y_mapped คือ: ปกติ=0, เสี่ยง=1, สงสัยป่วย=2
    # เราจะสร้าง mapping dictionary เพื่อแปลงค่ากลับ
    prediction_labels = {
        0: 'ปกติ',
        1: 'เสี่ยง',
        2: 'สงสัยป่วย'
    }

    model = load_model()

    if model:
        # อินเทอร์เฟซผู้ใช้
        st.subheader("Risk Assessment Form")
        st.write("ใส่ข้อมูลสุขภาพของคุณ เพื่อประเมินความเสี่ยงโรคเบาหวาน (DM)")

        # Create input form - Ensure input names match the features used for training
        # Based on X_cleaned columns from cell 0nWeTsnKDoQs:
        # 'age', 'sex', 'bmi', 'waist', 'systolic_bp', 'diastolic_bp', 'cholesterol'
        age_input = st.number_input("อายุ (ปี)", min_value=1, max_value=120, value=50)
        # sex was encoded as 0 for ชาย, 1 for หญิง in cell 0nWeTsnKDoQs
        sex_input_option = st.selectbox("เพศ", ["ชาย", "หญิง"])
        sex_encoded = 0 if sex_input_option == "ชาย" else 1 # Encode sex input
        bmi_input = st.number_input("ค่าดัชนีมวลกาย (BMI)", min_value=10.0, max_value=50.0, value=25.0)
        waist_input = st.number_input("รอบเอว (cm)", min_value=30, max_value=200, value=85)
        systolic_bp_input = st.number_input("ค่าความดันโลหิตตัวบน (Systolic)", min_value=80, max_value=250, value=130)
        diastolic_bp_input = st.number_input("ค่าความดันโลหิตตัวล่าง (Diastolic)", min_value=40, max_value=150, value=80) # Assuming diastolic_bp was also a feature
        cholesterol_input = st.number_input("ระดับไขมันรวม (Cholesterol)", min_value=100, max_value=400, value=180)
        # Note: 'glucose' was dropped in cell 0nWeTsnKDoQs before training the multi-class model.
        # If you need to include glucose, you'll need to retrain the model with it.

        # Create a DataFrame with the exact column names as the training data.
        # These column names are based on X_cleaned from cell 0nWeTsnKDoQs.
        user_input_dict = {
            'age': age_input,
            'sex': sex_encoded,
            'bmi': bmi_input,
            'waist': waist_input,
            'systolic_bp': systolic_bp_input,
            'diastolic_bp': diastolic_bp_input,
            'cholesterol': cholesterol_input
        }

        user_input_df = pd.DataFrame([user_input_dict])

        # ทำนาย
        if st.button("ประเมินความเสี่ยง"):
            # Ensure the order of columns in user_input_df matches the training data
            # Get feature names from the trained model if possible, or rely on the known training features
            # model.feature_names_in_ can be used in newer XGBoost versions
            # For compatibility, we'll rely on the known training features from cell 0nWeTsnKDoQs
            # training_feature_names = ['age', 'sex', 'bmi', 'waist', 'systolic_bp', 'diastolic_bp', 'cholesterol']

            # Ensure the columns in user_input_df are in the correct order as expected by the model
            # This requires knowing the exact order the model was trained on.
            # A safer approach is to save the feature names during training.
            # For now, we rely on the order observed in X_cleaned from cell 0nWeTsnKDoQs.
            try:
                # Attempt to get feature names from the model (works in newer XGBoost versions)
                if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                    expected_features = model.feature_names_in_
                else:
                    # Fallback to the manually defined training_feature_names if not available
                    # Ensure this matches the exact order from training!
                    expected_features = ['age', 'sex', 'bmi', 'waist', 'systolic_bp', 'diastolic_bp', 'cholesterol'] # Manually confirmed order

                user_input_ordered = user_input_df[expected_features]

                pred_encoded = model.predict(user_input_ordered)[0]
                prob_all_classes = model.predict_proba(user_input_ordered)[0] # Probabilities for all classes

                # Convert encoded prediction back to original label
                predicted_label = prediction_labels.get(pred_encoded, f"Unknown encoded label: {pred_encoded}")

                st.write(f"ผลการประเมินความเสี่ยง: **{predicted_label}**")
                # Optionally display probabilities for all classes
                st.write("ความน่าจะเป็นของแต่ละคลาส:")
                for class_id, prob in enumerate(prob_all_classes):
                    label = prediction_labels.get(class_id, f"Class {class_id}")
                    st.write(f"- {label}: {prob:.2f}")


            except KeyError as e:
                st.error(f"Feature mismatch: {e}. Ensure input columns match model training features.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        st.write("⚠️ ข้อมูลนี้ใช้เพื่อการประเมินเบื้องต้นเท่านั้น ควรพบแพทย์เพื่อการวินิจฉัย")

    else:
        st.write("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบและรันเซลล์ฝึกโมเดลก่อนครับ")


