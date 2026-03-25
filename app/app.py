import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")

# ---------- LOAD MODELS ----------

model = joblib.load("models/final_churn_model.pkl")
scaler = joblib.load("models/final_scaler.pkl")
features = joblib.load("models/final_features.pkl")

explainer = shap.TreeExplainer(model)

# ---------- REQUIRED RAW COLUMNS ----------

required_columns = [
"Age",
"Tenure in Months",
"Monthly Charge",
"Satisfaction Score",
"Number of Referrals",
"Internet Type",
"Contract",
"Paperless Billing"
]

# ---------- HEADER ----------

st.title("Customer Churn Intelligence System")
st.caption("AI-Powered Retention Analytics Dashboard")

c1,c2,c3 = st.columns(3)

c1.metric("Model","Tuned XGBoost")
c2.metric("Accuracy","96%")
c3.metric("Recall","94%")

st.divider()

# ---------- MODE SELECT ----------

mode = st.radio(
"Select Prediction Mode",
["Single Customer Prediction",
 "Bulk Dataset Prediction"]
)

st.divider()

###########################################################
# SINGLE CUSTOMER MODE
###########################################################

if mode == "Single Customer Prediction":

    defaults = {
    "age":40,
    "tenure":12,
    "monthly":60,
    "satisfaction":3,
    "referrals":0
    }

    for k,v in defaults.items():

        if k not in st.session_state:

            st.session_state[k]=v
            st.session_state[k+"_slider"]=v


    def sync_slider(name):
        st.session_state[name]=st.session_state[name+"_slider"]


    def sync_box(name):
        st.session_state[name+"_slider"]=st.session_state[name]


    st.header("Customer Profile")

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Customer Metrics")

        st.number_input("Age",18,90,key="age",on_change=sync_box,args=("age",))
        st.slider("",18,90,key="age_slider",on_change=sync_slider,args=("age",))

        st.number_input("Tenure in Months",0,72,key="tenure",on_change=sync_box,args=("tenure",))
        st.slider("",0,72,key="tenure_slider",on_change=sync_slider,args=("tenure",))

        st.number_input("Monthly Charge",10,120,key="monthly",on_change=sync_box,args=("monthly",))
        st.slider("",10,120,key="monthly_slider",on_change=sync_slider,args=("monthly",))

        st.number_input("Satisfaction Score",1,5,key="satisfaction",on_change=sync_box,args=("satisfaction",))
        st.slider("",1,5,key="satisfaction_slider",on_change=sync_slider,args=("satisfaction",))

        st.number_input("Number of Referrals",0,10,key="referrals",on_change=sync_box,args=("referrals",))
        st.slider("",0,10,key="referrals_slider",on_change=sync_slider,args=("referrals",))


    with col2:

        st.subheader("Service Options")

        fiber=st.radio("Fiber Internet",["No","Yes"])
        contract_two=st.radio("Two Year Contract",["No","Yes"])
        paperless=st.radio("Paperless Billing",["No","Yes"])


    predict=st.button("Predict Customer Risk")


    def draw_gauge(value):

        fig=go.Figure(go.Indicator(

        mode="gauge+number",

        value=value,

        number={'suffix':"%",'font':{'size':60}},

        gauge={

        'axis':{'range':[0,100]},

        'bar':{
        'color':"#00D4FF",
        'thickness':0.30
        },

        'steps':[

        {'range':[0,30],'color':"#1a7f4b"},
        {'range':[30,70],'color':"#d39c00"},
        {'range':[70,100],'color':"#c40000"}
        ]

        }

        ))

        fig.update_layout(
        height=420,
        paper_bgcolor="#0b1220",
        font=dict(color="white")
        )

        return fig


    if predict:

        input_data=pd.DataFrame(
        np.zeros((1,len(features))),
        columns=features
        )


        input_data["Age"]=st.session_state.age
        input_data["Tenure in Months"]=st.session_state.tenure
        input_data["Monthly Charge"]=st.session_state.monthly
        input_data["Satisfaction Score"]=st.session_state.satisfaction
        input_data["Number of Referrals"]=st.session_state.referrals


        if fiber=="Yes":
            input_data["Internet Type_Fiber Optic"]=1

        if contract_two=="Yes":
            input_data["Contract_Two Year"]=1

        if paperless=="Yes":
            input_data["Paperless Billing_Yes"]=1


        scaled=scaler.transform(input_data)

        prob=model.predict_proba(scaled)[0][1]

        risk=int(prob*100)


        # ---------- STRATEGY FIX (IMPORTANT) ----------

        if st.session_state.satisfaction<=2:
            strategy="Immediate customer support intervention"

        elif st.session_state.tenure<12:
            strategy="Provide onboarding retention discount"

        elif st.session_state.monthly>80:
            strategy="Offer cost reduction plan"

        elif st.session_state.referrals==0:
            strategy="Encourage referral program"

        else:
            strategy="Standard retention monitoring"



        st.divider()

        center = st.columns([1,3,1])[1]

        with center:

            st.header("Churn Risk Score")

            holder = st.empty()

            delay = 3.5/max(risk,1)

            for i in range(risk+1):

                holder.plotly_chart(
                    draw_gauge(i),
                    use_container_width=True
                )

                time.sleep(delay)


            st.markdown("<br>", unsafe_allow_html=True)


            if prob > 0.30:

                st.markdown(
                """
                <div style="
                background:#4b1d1d;
                padding:20px;
                border-radius:12px;
                text-align:center;
                font-size:26px;
                font-weight:700;
                color:white;
                ">
                ⚠ HIGH CHURN RISK DETECTED
                </div>
                """,
                unsafe_allow_html=True
                )

            else:

                st.markdown(
                """
                <div style="
                background:#184d2f;
                padding:20px;
                border-radius:12px;
                text-align:center;
                font-size:26px;
                font-weight:700;
                color:white;
                ">
                ✓ LOW RISK CUSTOMER
                </div>
                """,
                unsafe_allow_html=True
                )


            st.markdown("<br>", unsafe_allow_html=True)


            st.subheader("Recommended Retention Strategy")

            st.markdown(
            f"""
            <div style="
            background:#1f3a52;
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:24px;
            color:#4aa3ff;
            font-weight:600;
            ">
            {strategy}
            </div>
            """,
            unsafe_allow_html=True
            )


###########################################################
# BULK MODE (UNCHANGED)
###########################################################

else:

    st.header("Upload Customer Dataset")

    st.write("Dataset must include these columns:")
    st.write(required_columns)

    file=st.file_uploader("Upload CSV",type="csv")

    if file is not None:

        df=pd.read_csv(file)

        missing_cols=[col for col in required_columns if col not in df.columns]

        if len(missing_cols)>0:

            st.error("Dataset missing required columns:")
            st.write(missing_cols)

            st.stop()

        st.success("Dataset validated successfully")

        st.subheader("Preview")
        st.dataframe(df.head())


        df_encoded=pd.get_dummies(df)

        X=df_encoded.reindex(columns=features,fill_value=0)

        X_scaled=scaler.transform(X)

        probs=model.predict_proba(X_scaled)[:,1]

        df["Churn Probability"]=probs
        df["High Risk"]=df["Churn Probability"]>=0.30


        def strategy(row):

            if row["Satisfaction Score"]<=2:
                return "Support Intervention"

            elif row["Tenure in Months"]<12:
                return "Onboarding Discount"

            elif row["Monthly Charge"]>80:
                return "Cost Reduction Plan"

            elif row["Number of Referrals"]==0:
                return "Referral Program"

            else:
                return "Standard Monitoring"


        df["Strategy"]=df.apply(strategy,axis=1)


        st.header("Business KPIs")

        k1,k2,k3,k4=st.columns(4)

        k1.metric("Customers",len(df))
        k2.metric("High Risk",df["High Risk"].sum())
        k3.metric("High Risk %",round(df["High Risk"].mean()*100,1))
        k4.metric("Avg Churn Probability",round(df["Churn Probability"].mean(),2))


        st.divider()

        c1,c2=st.columns(2)

        with c1:

            st.subheader("Risk Distribution")

            fig=px.pie(df,names="High Risk")

            st.plotly_chart(fig)


        with c2:

            st.subheader("Strategy Distribution")

            fig2=px.bar(df["Strategy"].value_counts())

            st.plotly_chart(fig2)


        st.divider()

        st.header("Predictions")

        st.dataframe(df)

        st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        file_name="churn_predictions.csv"
        )