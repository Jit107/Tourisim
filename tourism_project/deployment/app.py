import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_repo_id = "jit0107/tourism_customer_predictor"
model_filename = "best_tourism_customer_predictor_v1.joblib"
model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Purchase Predictor")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the customer details and interaction data below to get a prediction.
""")

# User input for customer details and interaction data
st.header("Customer Details")
age = st.slider("Age", min_value=18, max_value=80, value=35)
typeofcontact_mapping = {"Company Invited": 0, "Self Inquiry": 1}
typeofcontact_display = st.selectbox("Type of Contact", list(typeofcontact_mapping.keys()), index=0)
typeofcontact = typeofcontact_mapping[typeofcontact_display]
citytier = st.selectbox("City Tier", [1, 2, 3], index=0)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business", "Unemployed"], index=0)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferredpropertystar = st.slider("Preferred Property Star (1-5)", min_value=1, max_value=5, value=3)
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=1)
passport_mapping = {"Yes": 1, "No": 0}
passport_display = st.selectbox("Passport Holder?", list(passport_mapping.keys()), index=1)
passport = passport_mapping[passport_display]
owncar_mapping = {"Yes": 1, "No": 0}
owncar_display = st.selectbox("Owns a Car?", list(owncar_mapping.keys()), index=1)
owncar = owncar_mapping[owncar_display]
numberofchildrenvisiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "Junior Manager", "VP", "Director", "President", "CEO", "Associate"], index=0)
monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

st.header("Customer Interaction Data")
pitchsatisfactionscore = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], index=1)
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'ProductPitched': productpitched,
    'NumberOfFollowups': numberoffollowups,
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the Wellness Tourism Package" if prediction == 1 else "Customer will NOT purchase the Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
