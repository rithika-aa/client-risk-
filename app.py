import sqlite3
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Customer Churn App", layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAF7F2;
        color: #3A3A3A;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #7A5C8F;
        font-weight: 700;
    }

    h2, h3 {
        color: #6F7FAF;
        font-weight: 600;
    }

    p, label, span, div {
        color: #3A3A3A;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F3EEF8;
        padding: 8px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #FFFDF8;
        border-radius: 8px;
        color: #7A5C8F;
        padding: 10px 16px;
        border: 1px solid #E7DDF1;
    }

    .stTabs [aria-selected="true"] {
        background-color: #EADCF8;
        color: #5D4773;
        font-weight: 600;
    }

    .stButton > button {
        background-color: #DFA7C8;
        color: #3A3A3A;
        border-radius: 8px;
        border: 1px solid #D392B8;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #CDB4DB;
        color: #2F2F2F;
        border: 1px solid #BFA0D1;
    }

    [data-testid="stMetric"] {
        background-color: #FFFDF8;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #E6E0D8;
        box-shadow: 0 2px 8px rgba(122, 92, 143, 0.08);
    }

    [data-testid="stDataFrame"] {
        background-color: #FFFDF8;
        border-radius: 8px;
        border: 1px solid #E6E0D8;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFDF8;
        border-radius: 8px;
    }

    .stAlert {
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] {
        background-color: #F3EEF8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            support_calls INTEGER,
            contract TEXT,
            payment_method TEXT,
            internet_service TEXT,
            tech_support TEXT,
            online_security TEXT,
            prediction TEXT,
            probability REAL
        )
        """
    )

    conn.commit()
    conn.close()


def register_user(username, password):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users VALUES (?, ?)",
            (username, hash_password(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password)),
    )

    user = cursor.fetchone()
    conn.close()

    return user is not None


def save_prediction(
    username,
    tenure,
    monthly_charges,
    total_charges,
    support_calls,
    contract,
    payment_method,
    internet_service,
    tech_support,
    online_security,
    prediction,
    probability,
):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            username, date, tenure, monthly_charges, total_charges,
            support_calls, contract, payment_method, internet_service,
            tech_support, online_security, prediction, probability
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tenure,
            monthly_charges,
            total_charges,
            support_calls,
            contract,
            payment_method,
            internet_service,
            tech_support,
            online_security,
            prediction,
            probability,
        ),
    )

    conn.commit()
    conn.close()


def get_predictions(username):
    conn = sqlite3.connect("app_data.db")

    predictions = pd.read_sql_query(
        "SELECT * FROM predictions WHERE username = ? ORDER BY id DESC",
        conn,
        params=(username,),
    )

    conn.close()

    return predictions


@st.cache_resource(show_spinner=True)
def train_model():
    df = pd.read_csv("customer_churn_dataset.csv")
    df["internet_service"] = df["internet_service"].fillna("No internet")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    categorical = [
        "contract",
        "payment_method",
        "internet_service",
        "tech_support",
        "online_security",
    ]

    numerical = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "support_calls",
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    model.fit(X, y)

    return model, df


init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


model, df = train_model()
df["internet_service"] = df["internet_service"].fillna("No internet")


st.title("Customer Churn Prediction Dashboard")

if st.session_state.logged_in:
    st.success(f"Logged in as {st.session_state.username}")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Home",
        "Login",
        "Predict Customer",
        "Saved Predictions",
        "Dataset Insights",
        "Model Insights",
    ]
)


with tab0:
    st.header("Welcome")

    st.write(
        "This app predicts whether a customer is likely to leave or stay. "
        "You can enter customer details, view churn probability, and save predictions "
        "after logging in."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(df))

    with col2:
        churn_rate = df["churn"].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        st.metric("Features Used", "9")

    st.info("Go to the Predict Customer tab to start making predictions.")


with tab1:
    st.header("Login or Register")

    if st.session_state.logged_in:
        st.write(f"You are currently logged in as **{st.session_state.username}**.")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.success("You have been logged out.")

    else:
        choice = st.radio("Choose an option", ["Login", "Register"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if choice == "Register":
            if st.button("Create Account"):
                if username == "" or password == "":
                    st.error("Please enter a username and password.")
                elif register_user(username, password):
                    st.success("Account created. You can now log in.")
                else:
                    st.error("Username already exists.")

        if choice == "Login":
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                else:
                    st.error("Invalid username or password.")


with tab2:
    st.header("Predict Customer")

    st.write("Enter customer details below to predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure", 0, 200, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 100000.0, 0.00)
        total_charges = st.number_input("Total Charges", 0.0, 50000.0, 0.00)
        support_calls = st.number_input("Support Calls", 0, 50, 0)

    with col2:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            ["Credit", "Debit", "Cash", "UPI"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber", "No internet"],
        )

        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])

    if st.button("Predict"):
        new_customer = pd.DataFrame(
            {
                "tenure": [tenure],
                "monthly_charges": [monthly_charges],
                "total_charges": [total_charges],
                "contract": [contract],
                "payment_method": [payment_method],
                "internet_service": [internet_service],
                "tech_support": [tech_support],
                "online_security": [online_security],
                "support_calls": [support_calls],
            }
        )

        prediction = model.predict(new_customer)[0]
        probability = model.predict_proba(new_customer)[0][1]

        result = "Leave" if prediction == 1 else "Stay"

        st.metric("Churn Probability", f"{probability * 100:.1f}%")

        if prediction == 1:
            st.error("This customer is likely to leave.")
        else:
            st.success("This customer is likely to stay.")

        if st.session_state.logged_in:
            save_prediction(
                st.session_state.username,
                tenure,
                monthly_charges,
                total_charges,
                support_calls,
                contract,
                payment_method,
                internet_service,
                tech_support,
                online_security,
                result,
                probability,
            )

            st.success("Prediction saved to your account.")
        else:
            st.warning("Login to save this prediction.")


with tab3:
    st.header("Saved Predictions")

    if st.session_state.logged_in:
        saved_data = get_predictions(st.session_state.username)

        if saved_data.empty:
            st.info("No saved predictions yet.")
        else:
            st.dataframe(saved_data, use_container_width=True)
    else:
        st.warning("Please log in to view your saved predictions.")


with tab4:
    st.header("Dataset Insights")

    st.caption(
        "This dashboard uses a customer dataset to explore patterns linked "
        "to customers leaving."
    )

    st.write(
        "Churn means the percentage of customers who discontinue their relationship "
        "with a company within a given time frame."
    )

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df["churn"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(
            churn_counts,
            labels=churn_counts.index,
            autopct="%1.1f%%",
            colors=["#FFBBCB", "#B0D03E"],
        )
        ax.set_title("Overall Churn Distribution")
        st.pyplot(fig)

    with col2:
        contract_churn = pd.crosstab(df["contract"], df["churn"])

        fig, ax = plt.subplots()
        contract_churn.plot(
            kind="bar",
            ax=ax,
            color=["#32D8E7DE", "#E63737"],
        )
        ax.set_title("Prediction by Contract Type")
        ax.set_xlabel("Contract")
        ax.set_ylabel("Number of Customers")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.write("Average values by churn status:")

    st.dataframe(
        df.groupby("churn")[
            ["tenure", "monthly_charges", "total_charges", "support_calls"]
        ].mean(),
        use_container_width=True,
    )


with tab5:
    st.header("Model Insights")

    st.caption(
        "Feature importance shows which inputs the model relied on most "
        "when making predictions."
    )

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color="#6B5E43",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Factors Affecting Churn")
    st.pyplot(fig)

    st.write(
        "In simple terms, the model mostly looks at charges, tenure, support calls, "
        "and contract type to decide whether a customer may leave."
    )

