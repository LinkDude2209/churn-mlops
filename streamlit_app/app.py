\
import os
import requests
import streamlit as st

# ----------------------------------
# CONFIGURATION
# ----------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# HEADER
# ----------------------------------

st.title("📊 Churn Intelligence")
st.caption(
    "Telecom Customer Retention | ML-powered prediction"
)

st.divider()

# ----------------------------------
# MODEL STATUS
# ----------------------------------

try:
    health_url = API_URL.rsplit("/", 1)[0] + "/health"
    health = requests.get(health_url, timeout=3)

    if health.ok:
        st.success("Prediction API is online")
    else:
        st.warning("Prediction API is responding with an error")

except requests.exceptions.RequestException:
    st.error("Prediction API is offline. Start FastAPI first.")

# ----------------------------------
# CUSTOMER FORM
# ----------------------------------

st.subheader("Customer Profile")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)", 0, 72, 12
        )

    with col2:
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        device = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    st.subheader("Billing Information")

    col4, col5 = st.columns(2)

    with col4:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col5:
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

# ----------------------------------
# PREDICTION
# ----------------------------------

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        with st.spinner("Analyzing customer profile..."):

            response = requests.post(
                API_URL,
                json=customer,
                timeout=20
            )

            response.raise_for_status()
            result = response.json()

        prediction = result["prediction"]
        churn = result["churn_probability"]
        retention = result["no_churn_probability"]

        st.divider()
        st.header("Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Churn Probability",
                f"{churn * 100:.2f}%"
            )

        with c2:
            st.metric(
                "Retention Probability",
                f"{retention * 100:.2f}%"
            )

        with c3:
            st.metric(
                "Predicted Outcome",
                prediction
            )

        if prediction == "Churn":
            st.error(
                "⚠️ High-risk customer: Consider a retention strategy."
            )
        else:
            st.success(
                "✅ Customer predicted to stay."
            )

        st.subheader("Probability Distribution")

        st.progress(churn)

        st.write({
            "Churn": f"{churn * 100:.2f}%",
            "No Churn": f"{retention * 100:.2f}%"
        })

        with st.expander("View submitted customer data"):
            st.json(customer)

        with st.expander("View raw API response"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Check the API server.")

    except requests.exceptions.RequestException as error:
        st.error(f"Prediction request failed: {error}")

    except Exception as error:
        st.error(f"Unexpected error: {error}")

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "Churn Intelligence | Random Forest | "
    "Scikit-learn + FastAPI + Streamlit"
)
