from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Define paths
model_path = '/Users/arjunkhatiwada/Documents/master_project_code/mp_cp_ak/voting_classifier_rf_xgb.pkl'
scaler_path = '/Users/arjunkhatiwada/Documents/master_project_code/mp_cp_ak/scaler.pkl'

# Load the trained model and scaler
# voting_classifier_model = joblib.load(model_path)
# with open(model_path, 'rb') as file:
#     voting_classifier_model = pickle.load(file)

with open(model_path, 'rb') as file:
    voting_classifier_model = pickle.load(file)

# with open(scaler_path, 'rb') as file:
#     loaded_scaler = pickle.load(file)

loaded_scaler = joblib.load(scaler_path)

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # To store prediction results
    if request.method == 'POST':
        try:
            # Get data from form
            user_input_data = {
                'Tenure': int(request.form['Tenure']),
                'SatisfactionScore': int(request.form['SatisfactionScore']),
                'Complain': int(request.form['Complain']),
                'DaySinceLastOrder': int(request.form['DaySinceLastOrder']),
                'CouponUsed': int(request.form['CouponUsed']),
                'WarehouseToHome': int(request.form['WarehouseToHome']),
                'NumberOfAddress': int(request.form['NumberOfAddress']),
                'OrderCount': int(request.form['OrderCount']),
                'CityTier': int(request.form['CityTier']),
                'MaritalStatus': request.form['MaritalStatus'],
                'PreferredLoginDevice': request.form['PreferredLoginDevice'],
                'PreferredPaymentMode': request.form['PreferredPaymentMode'],
                'Gender': request.form['Gender'],
                'PreferedOrderCat': request.form['PreferedOrderCat']
            }

            # Create DataFrame from user input
            df_test = pd.DataFrame(user_input_data, index=[0])
            
            # One-hot encode for categorical variables
            categorical_df_test = df_test.select_dtypes(include=['object'])
            df_test = pd.get_dummies(df_test, columns=list(categorical_df_test.columns), drop_first=False)
            
            # Ensure all expected features are present
            X_train_cols = [
                'Tenure', 'CityTier', 'WarehouseToHome', 'SatisfactionScore',
                'NumberOfAddress', 'Complain', 'CouponUsed', 'OrderCount',
                'DaySinceLastOrder', 'PreferredLoginDevice_Mobile Phone',
                'PreferredLoginDevice_Phone', 'PreferredPaymentMode_COD',
                'PreferredPaymentMode_Cash on Delivery',
                'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
                'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
                'Gender_Male', 'PreferedOrderCat_Grocery',
                'PreferedOrderCat_Laptop & Accessory', 'PreferedOrderCat_Mobile',
                'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others',
                'MaritalStatus_Married', 'MaritalStatus_Single'
            ]

            # Ensure all columns expected by the model are present in the DataFrame
            for col in X_train_cols:
                if col not in df_test.columns:
                    df_test[col] = 0

            # Reorder columns to match the order used during training
            df_test = df_test[X_train_cols]

            # Store the original input values for suggestions
            original_values = user_input_data.copy()

            # Scale the input data
            scaled_row = loaded_scaler.transform(df_test)
            user_df = pd.DataFrame(scaled_row, columns=X_train_cols)

            # Use the loaded model to predict churn
            y_pred = voting_classifier_model.predict(user_df)
            y_proba = voting_classifier_model.predict_proba(user_df)

            # Prepare output
            churn_status = "Likely to Churn" if y_pred[0] == 1 else "Likely to Not Churn"
            churn_probability = y_proba[0][1] * 100  # Convert to percentage

            # Prepare dynamic suggestions based on original user input
            suggestions = []

            if y_pred[0] == 1:  # Only provide suggestions if likely to churn
                # Use original input values instead of scaled values
                tenure = original_values['Tenure']
                satisfaction_score = original_values['SatisfactionScore']
                complain = original_values['Complain']
                day_since_last_order = original_values['DaySinceLastOrder']
                coupon_used = original_values['CouponUsed']
                warehouse_to_home = original_values['WarehouseToHome']
                number_of_address = original_values['NumberOfAddress']
                order_count = original_values['OrderCount']
                city_tier = original_values['CityTier']

                # Define suggestions based on conditions using original values
                if tenure < 3:
                    suggestions.append("Engage the customer through targeted loyalty programs to increase their tenure.")
                if satisfaction_score < 3:
                    suggestions.append("Conduct a survey to gather feedback and improve customer satisfaction.")
                if complain == 1:
                    suggestions.append("Implement a proactive outreach strategy for customers with complaints to resolve issues quickly.")
                if day_since_last_order > 60:
                    suggestions.append("Send reminders or incentives to encourage more frequent orders from customers.")
                if coupon_used > 3:
                    suggestions.append("Analyze coupon usage patterns; consider offering personalized promotions to high-usage customers.")
                if warehouse_to_home > 10:  # Assuming 10 is a threshold for long delivery distance
                    suggestions.append("Evaluate delivery times and explore options to enhance the efficiency of the delivery process.")
                if number_of_address < 2:
                    suggestions.append("Encourage customers to update their addresses for better service delivery options.")
                if order_count < 3:
                    suggestions.append("Increase engagement with targeted marketing campaigns based on order history.")

                # Suggestions based on city tier values without hierarchy
                if city_tier == 1:
                    suggestions.append("Offer exclusive deals for customers in Tier 1 cities to attract them.")
                elif city_tier == 2:
                    suggestions.append("Focus on improving service delivery and customer support in Tier 2 cities.")
                elif city_tier == 3:
                    suggestions.append("Consider localized marketing strategies and community engagement for Tier 3 cities.")

            # Create result dictionary to pass to template
            result = {
                'churn_status': churn_status,
                'churn_probability': f"{churn_probability:.2f}%",
                'input_data': original_values,  # Use original values here for display
                'suggestions': suggestions
            }

        except KeyError as e:
            print(f"Missing key in form data: {e}")
            return "Form data is missing required fields.", 400

    return render_template('index.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)