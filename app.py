import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Will your customer leave?", layout="centered")

st.write("This model predicts whether your client will leave or stay. Uses probability to give the best results")

model = joblib.load("customer_churn_model.pkl")
df = pd.read_csv("customer_churn_dataset.csv")
df["internet_service"] = df["internet_service"].fillna("No internet")

st.title("Customer Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["Predict Customer", "Dataset Insights", "Model Insights"])

with tab1:
    st.subheader("Enter your Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure", 0, 200, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 100000.0, 0.00)
        total_charges = st.number_input("Total Charges", 0.0, 50000.0, 0.00)
        support_calls = st.number_input("Support Calls", 0, 50, 0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Credit", "Debit", "Cash", "UPI"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "No internet"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])

    if st.button("Predict"):
        new_customer = pd.DataFrame({
            "tenure": [tenure],
            "monthly_charges": [monthly_charges],
            "total_charges": [total_charges],
            "contract": [contract],
            "payment_method": [payment_method],
            "internet_service": [internet_service],
            "tech_support": [tech_support],
            "online_security": [online_security],
            "support_calls": [support_calls]
        })

        prediction = model.predict(new_customer)[0]
        probability = model.predict_proba(new_customer)[0][1]

        st.metric("Churn Probability", f"{probability * 100:.1f}%")

        if prediction == 1:
            st.error("This customer is likely to leave.")
        else:
            st.success("This customer is likely to stay.")

with tab2:
    st.subheader("Dataset Insights and values")
    st.write("churn: percentage of customers who discontinue their relationship with a company within a given time frame")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df["churn"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", colors=["#C2185B", "#8E24AA"])
        ax.set_title("Overall Churn Distribution")
        st.pyplot(fig)

    with col2:
        contract_churn = pd.crosstab(df["contract"], df["churn"])
        fig, ax = plt.subplots()
        contract_churn.plot(kind="bar", ax=ax,color=["#C2185B", "#8E24AA"])
        ax.set_title("prediction by Contract Type")
        ax.set_xlabel("Contract")
        ax.set_ylabel("Number of Customers")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.write("Average values by predicting the status:")
    st.dataframe(
        df.groupby("churn")[["tenure", "monthly_charges", "total_charges", "support_calls"]].mean()
    )

with tab3:
    st.subheader("What Affects this Most?")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(feature_importance["feature"], feature_importance["importance"],color="#AD1457")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Factors Affecting Churn")
    st.pyplot(fig)

    st.write(
        "In simple terms, the model mostly looks at charges, tenure, support calls, "
        "and contract type to decide whether a customer may leave."
    )

