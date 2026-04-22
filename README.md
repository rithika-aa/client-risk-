# Client Risk Analysis

Client Risk Analysis is an interactive machine learning dashboard built with Streamlit. The app predicts whether a customer is likely to churn based on customer details such as tenure, monthly charges, total charges, contract type, support calls, and service information.

## Live App

[View the deployed app](https://clientriskmodel.streamlit.app/)

## Project Overview

Customer churn means a customer stops using a company’s product or service. This project helps analyze customer risk by predicting whether a customer is likely to stay or leave.

The dashboard allows users to enter customer information, generate churn predictions, view their input data, and explore visual insights from both the original dataset and the machine learning model.

## Features

- Predict customer churn probability
- Add multiple customer inputs during a session
- View graphs based only on user-entered data
- Explore original dataset insights
- View model feature importance
- Interactive navigation inside the app
- Clean white dashboard design with black and yellow accents
- Plotly charts with readable labels, legends, and graph text

## Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Random Forest Classifier
- GitHub
- Streamlit Community Cloud

## Machine Learning Model

The app uses a Scikit-learn pipeline with:

- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `RandomForestClassifier` for churn prediction


The model predicts:

```txt
1 = Likely to churn
0 = Likely to stay
```

The model uses the following customer details:
```txt
tenure
monthly_charges
total_charges
support_calls
contract
payment_method
internet_service
tech_support
online_security
```
PROJECT STRUCTURE
```txt
client-risk-/
│
├── app.py
├── customer_churn_dataset.csv
├── requirements.txt
├── README.md
└── churn.ipynb
```

## How to Run Locally
Clone the repository:
```txt
git clone https://github.com/rithika-aa/client-risk-.git
cd client-risk-
```
Install dependencies:
```txt
pip install -r requirements.txt
```
Run the app:
```txt
python -m streamlit run app.py
```
Your `requirements.txt` must include:
```txt
streamlit
pandas
plotly
scikit-learn
```
## App Sections

## Home
Introduces the dashboard and shows quick project metrics.

## Predict
Allows users to enter customer details and predict churn probability.

## My Data
Displays the customer inputs added during the current session and creates graphs based only on that user-entered data.

## Dataset
Shows visual insights from the original dataset, including churn distribution and churn by contract type.

## Model
Shows the top features that influence the machine learning model’s churn predictions.

## Example Prediction

Example high-risk customer input:
```txt
Tenure: 2
Monthly Charges: 95
Total Charges: 190
Support Calls: 8
Contract: Month-to-month
Payment Method: Debit
Internet Service: Fiber
Tech Support: No
Online Security: No
```
## Future Improvements

Add secure user login
Save user predictions with an online database
Add model accuracy metrics
Add confusion matrix and classification report
Allow CSV upload for batch predictions
Add downloadable prediction reports
Improve mobile responsiveness

## Author
Created by Rithika

## License
This project is for educational and portfolio purposes.
