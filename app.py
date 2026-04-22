import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="Client Risk Analysis",
    layout="wide",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #FFFFFF;
        color: #111111;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1280px;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .top-navbar {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 14px;
        padding: 18px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.07);
    }

    .brand-title {
        font-family: 'Playfair Display', serif;
        font-size: 36px;
        font-weight: 700;
        color: #111111 !important;
        white-space: nowrap;
    }

    .brand-accent {
        color: #D9A900 !important;
    }

    h1, h2, h3 {
        color: #111111 !important;
        font-weight: 800;
    }

    p, label, span, div {
        color: #111111;
    }

    .hero-panel {
        background: linear-gradient(135deg, #111111 0%, #2A2A2A 100%);
        border-radius: 18px;
        padding: 38px;
        margin-bottom: 30px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
    }

    .hero-kicker {
        color: #F5C542 !important;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #FFFFFF !important;
        font-size: 52px;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 14px;
    }

    .hero-copy {
        color: #F5F5F5 !important;
        font-size: 18px;
        line-height: 1.6;
        max-width: 900px;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #E8E8E8;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.07);
        max-width: 270px;
        min-height: 90px;
    }

    [data-testid="stMetricLabel"] {
        color: #111111 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111111 !important;
        font-size: 30px !important;
        font-weight: 800 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #111111 0%, #333333 100%);
        color: #FFFFFF !important;
        border-radius: 10px;
        border: 1px solid #111111;
        padding: 0.68rem 1.2rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.14);
    }

    .stButton > button:hover {
        background: #D9A900;
        color: #FFFFFF !important;
        border: 1px solid #D9A900;
    }

    .stButton > button p {
        color: #FFFFFF !important;
    }

    .stTextInput input,
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #DADADA !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #DADADA !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #111111 !important;
    }

    ul[role="listbox"],
    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    li[role="option"]:hover {
        background-color: #FFF4BF !important;
    }

    .stAlert {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


POSITIVE_CHURN_VALUES = [
    "1",
    "yes",
    "true",
    "leave",
    "left",
    "churn",
    "churned",
]


def convert_churn_to_number(churn_column):
    return churn_column.astype(str).str.lower().isin(POSITIVE_CHURN_VALUES).astype(int)


def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111111", size=14),
        title=dict(font=dict(color="#111111", size=20)),
        legend=dict(font=dict(color="#111111", size=13)),
        xaxis=dict(
            title_font=dict(color="#111111"),
            tickfont=dict(color="#111111"),
            gridcolor="#E6E6E6",
            zerolinecolor="#CCCCCC",
        ),
        yaxis=dict(
            title_font=dict(color="#111111"),
            tickfont=dict(color="#111111"),
            gridcolor="#E6E6E6",
            zerolinecolor="#CCCCCC",
        ),
    )

    return fig


@st.cache_resource(show_spinner=True)
def train_model():
    df = pd.read_csv("customer_churn_dataset.csv")
    df["internet_service"] = df["internet_service"].fillna("No internet")
    df["churn_number"] = convert_churn_to_number(df["churn"])

    X = df.drop(["churn", "churn_number"], axis=1)
    y = df["churn_number"]

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


model, df = train_model()

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "user_predictions" not in st.session_state:
    st.session_state.user_predictions = []


def go_to(page_name):
    st.session_state.page = page_name


st.markdown(
    """
    <div class="top-navbar">
        <div class="brand-title">Client Risk <span class="brand-accent">Analysis</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4, nav5 = st.columns(5)

with nav1:
    if st.button("Home", key="nav_home"):
        go_to("Home")

with nav2:
    if st.button("Predict", key="nav_predict"):
        go_to("Predict")

with nav3:
    if st.button("My Data", key="nav_my_data"):
        go_to("My Data")

with nav4:
    if st.button("Dataset", key="nav_dataset"):
        go_to("Dataset Insights")

with nav5:
    if st.button("Model", key="nav_model"):
        go_to("Model Insights")


if st.session_state.page == "Home":
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Machine Learning Dashboard</div>
            <div class="hero-title">Analyze customer risk before they leave.</div>
            <div class="hero-copy">
                Client Risk Analysis predicts customer churn using customer behavior,
                billing patterns, support activity, and service details. Enter customer
                information, compare risk levels, and explore model insights through a
                clean interactive dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Dataset Customers", len(df))

    with col2:
        st.metric("Your Saved Inputs", len(st.session_state.user_predictions))

    st.subheader("Start exploring")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Start Prediction", key="home_start_prediction"):
            go_to("Predict")
            st.rerun()

    with col2:
        if st.button("View My Data", key="home_view_my_data"):
            go_to("My Data")
            st.rerun()

    with col3:
        if st.button("Explore Dataset", key="home_explore_dataset"):
            go_to("Dataset Insights")
            st.rerun()


elif st.session_state.page == "Predict":
    st.header("Predict Customer Churn")
    st.write("Enter customer details below. The app will predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        customer_name = st.text_input("Customer Name", "Customer 1")
        tenure = st.number_input("Tenure", 0, 200, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 100000.0, 0.0)
        total_charges = st.number_input("Total Charges", 0.0, 500000.0, 0.0)
        support_calls = st.number_input("Support Calls", 0, 50, 0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Credit", "Debit", "Cash", "UPI"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber", "No internet"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])

    if st.button("Predict and Add to My Data", key="predict_add_data"):
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
        result = "Likely to Leave" if prediction == 1 else "Likely to Stay"

        st.session_state.user_predictions.append(
            {
                "customer_name": customer_name,
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "support_calls": support_calls,
                "contract": contract,
                "payment_method": payment_method,
                "internet_service": internet_service,
                "tech_support": tech_support,
                "online_security": online_security,
                "prediction": result,
                "churn_probability": round(probability * 100, 2),
            }
        )

        st.success("Prediction added to your input data.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Churn Probability", f"{probability * 100:.1f}%")

            if prediction == 1:
                st.error("This customer is likely to leave.")
            else:
                st.success("This customer is likely to stay.")

        with col2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={"text": "Churn Risk", "font": {"color": "#111111"}},
                    number={"font": {"color": "#111111"}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#111111",
                            "tickfont": {"color": "#111111"},
                        },
                        "bar": {"color": "#D9A900"},
                        "steps": [
                            {"range": [0, 40], "color": "#F8E619"},
                            {"range": [40, 70], "color": "#9EF72A"},
                            {"range": [70, 100], "color": "#ED0A43"},
                        ],
                    },
                )
            )
            fig.update_layout(
                paper_bgcolor="#FFFFFF",
                font_color="#111111",
            )
            st.plotly_chart(fig, use_container_width=True)

    if st.button("Go to My Input Data", key="predict_go_my_data"):
        go_to("My Data")
        st.rerun()


elif st.session_state.page == "My Data":
    st.header("My Input Data")

    if len(st.session_state.user_predictions) == 0:
        st.info("No customer data has been added yet. Go to Predict Customer first.")

        if st.button("Add Customer Data", key="my_data_add_customer"):
            go_to("Predict")
            st.rerun()

    else:
        user_df = pd.DataFrame(st.session_state.user_predictions)

        st.subheader("Your Saved Inputs")
        st.dataframe(user_df, use_container_width=True)

        st.subheader("Graphs Based Only on Your Input Data")

        graph_choice = st.selectbox(
            "Choose a graph",
            [
                "Churn Probability by Customer",
                "Monthly Charges vs Churn Probability",
                "Support Calls vs Churn Probability",
                "Prediction Result Count",
            ],
        )

        chart_colors = {
            "Likely to Leave": "#FB4399B6",
            "Likely to Stay": "#D9A900",
        }

        if graph_choice == "Churn Probability by Customer":
            fig = px.bar(
                user_df,
                x="customer_name",
                y="churn_probability",
                color="prediction",
                color_discrete_map=chart_colors,
                title="Churn Probability by Customer",
            )

        elif graph_choice == "Monthly Charges vs Churn Probability":
            fig = px.scatter(
                user_df,
                x="monthly_charges",
                y="churn_probability",
                color="prediction",
                size="support_calls",
                hover_name="customer_name",
                color_discrete_map=chart_colors,
                title="Monthly Charges vs Churn Probability",
            )

        elif graph_choice == "Support Calls vs Churn Probability":
            fig = px.scatter(
                user_df,
                x="support_calls",
                y="churn_probability",
                color="prediction",
                hover_name="customer_name",
                color_discrete_map=chart_colors,
                title="Support Calls vs Churn Probability",
            )

        else:
            count_df = user_df["prediction"].value_counts().reset_index()
            count_df.columns = ["prediction", "count"]

            fig = px.pie(
                count_df,
                names="prediction",
                values="count",
                color="prediction",
                color_discrete_map=chart_colors,
                title="Prediction Result Count",
            )

        fig = style_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Explain This Graph", key="explain_user_graph"):
            avg_probability = user_df["churn_probability"].mean()
            highest_customer = user_df.sort_values(
                by="churn_probability",
                ascending=False,
            ).iloc[0]

            st.info(
                f"Your input data has an average churn probability of "
                f"{avg_probability:.1f}%. The highest-risk customer is "
                f"{highest_customer['customer_name']} with a churn probability of "
                f"{highest_customer['churn_probability']:.1f}%. Customers with higher "
                f"charges, more support calls, and shorter tenure may show higher churn risk."
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Add Another Customer", key="add_another_customer"):
                go_to("Predict")
                st.rerun()

        with col2:
            if st.button("Clear My Input Data", key="clear_my_input_data"):
                st.session_state.user_predictions = []
                st.success("Your input data has been cleared.")
                st.rerun()


elif st.session_state.page == "Dataset Insights":
    st.header("Dataset Insights")

    st.write(
        "This section uses the original dataset. The dataset table is hidden below unless "
        "you choose to show it. You can toggle it to the dataset used from kaggle."
    )

    dataset_colors = {
        "Stayed": "#106403",
        "Churned": "#CF2463",
    }

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df["churn_number"].value_counts().reset_index()
        churn_counts.columns = ["churn", "count"]
        churn_counts["churn"] = churn_counts["churn"].map(
            {0: "Stayed", 1: "Churned"}
        )

        fig = px.pie(
            churn_counts,
            names="churn",
            values="count",
            color="churn",
            color_discrete_map=dataset_colors,
            title="Original Dataset Churn Distribution",
        )

        fig = style_chart(fig)
        fig.update_traces(textfont=dict(color="#111111"))

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        contract_churn = (
            df.groupby(["contract", "churn_number"])
            .size()
            .reset_index(name="count")
        )

        contract_churn["churn"] = contract_churn["churn_number"].map(
            {0: "Stayed", 1: "Churned"}
        )

        fig = px.bar(
            contract_churn,
            x="contract",
            y="count",
            color="churn",
            barmode="group",
            color_discrete_map=dataset_colors,
            title="Churn by Contract Type",
        )

        fig = style_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

    if st.button("Explain Dataset Graphs", key="explain_dataset_graphs"):
        st.info(
            "These graphs summarize the original dataset used to train the model. "
            "The churn distribution shows how many customers stayed or left. "
            "The contract chart helps show whether certain contract types are linked "
            "with higher churn."
        )

    show_dataset = st.toggle("Show Original Dataset", key="show_original_dataset")

    if show_dataset:
        st.dataframe(df.drop(columns=["churn_number"]), use_container_width=True)


elif st.session_state.page == "Model Insights":
    st.header("Model Insights")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False).head(10)

    fig = px.bar(
        feature_importance,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#DBB700", "#F4D76B", "#6B7280"],
        title="Top 10 Factors Affecting Churn",
    )

    fig = style_chart(fig)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Explain Model Insights", key="explain_model_insights"):
        top_feature = feature_importance.iloc[0]["feature"]

        st.info(
            f"The model uses these features to make churn predictions. "
            f"The most important feature in this trained model is `{top_feature}`. "
            f"Higher feature importance means the model relied on that input more often "
            f"when deciding whether a customer may leave."
        )

