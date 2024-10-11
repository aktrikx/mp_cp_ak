from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model_path = '/Users/arjunkhatiwada/Documents/master_project_code/mp_cp_ak/random_forest_model_after_smote.pkl'
with open(model_path, 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # To store prediction results
    if request.method == 'POST':
        # Get data from form
        user_input_data = {
            'Tenure': request.form['Tenure'],
            'SatisfactionScore': request.form['SatisfactionScore'],
            'Complain': request.form['Complain'],
            'DaySinceLastOrder': request.form['DaySinceLastOrder'],
            'CouponUsed': request.form['CouponUsed'],
            'WarehouseToHome': request.form['WarehouseToHome'],
            'NumberOfAddress': request.form['NumberOfAddress'],
            'OrderCount': request.form['OrderCount'],
            'CityTier': request.form['CityTier'],
            'MaritalStatus': request.form['MaritalStatus'],
            'PreferredLoginDevice': request.form['PreferredLoginDevice'],
            'PreferredPaymentMode': request.form['PreferredPaymentMode'],
            'Gender': request.form['Gender'],
            'PreferedOrderCat': request.form['PreferedOrderCat']
        }

        # Step 3: Convert input data to a DataFrame
        user_df = pd.DataFrame([user_input_data])

        # Step 4: Transform categorical columns using one-hot encoding
        user_df = pd.get_dummies(user_df, columns=['MaritalStatus', 'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat'])

        # Step 5: Get the feature names from the model training
        trained_feature_names = loaded_rf_model.feature_names_in_

        # Step 6: Ensure all columns expected by the model are present in the DataFrame
        for col in trained_feature_names:
            if col not in user_df.columns:
                user_df[col] = 0

        # Reorder columns to match the order used during training
        user_df = user_df[trained_feature_names]

        # Step 7: Use the loaded model to predict churn
        y_pred = loaded_rf_model.predict(user_df)

        # Step 8: Get the probability of churn (percentage)
        y_proba = loaded_rf_model.predict_proba(user_df)

        # Prepare output
        churn_status = "Likely to Churn" if y_pred[0] == 1 else "Likely to Not Churn"
        churn_probability = y_proba[0][1] * 100  # Convert to percentage

        # Prepare suggestions if customer is likely to churn
        suggestions = []
        if y_pred[0] == 1:  # Customer is likely to churn
            actionable_suggestions = {
                'Tenure': "Engage the customer through targeted loyalty programs to increase their tenure.",
                'SatisfactionScore': "Conduct a survey to gather feedback and improve customer satisfaction.",
                'Complain': "Implement a proactive outreach strategy for customers with complaints to resolve issues quickly.",
                'DaySinceLastOrder': "Send reminders or incentives to encourage more frequent orders from customers.",
                'CouponUsed': "Analyze coupon usage patterns; consider offering personalized promotions to high-usage customers.",
                'WarehouseToHome': "Evaluate delivery times and explore options to enhance the efficiency of the delivery process.",
                'NumberOfAddress': "Encourage customers to update their addresses for better service delivery options.",
                'OrderCount': "Increase engagement with targeted marketing campaigns based on order history.",
                'CityTier': "Customize offers based on city tier to enhance relevance and customer engagement."
            }

            # Check feature importance and user input dynamically
            for feature in actionable_suggestions.keys():
                if feature in trained_feature_names:
                    # You can modify conditions based on user_df values to be more specific
                    if feature == 'SatisfactionScore' and float(user_df['SatisfactionScore'].values[0]) < 3:
                        suggestions.append(actionable_suggestions[feature])
                    elif feature == 'Complain' and int(user_df['Complain'].values[0]) == 1:
                        suggestions.append(actionable_suggestions[feature])
                    elif feature == 'DaySinceLastOrder' and int(user_df['DaySinceLastOrder'].values[0]) > 60:
                        suggestions.append(actionable_suggestions[feature])
                    elif feature == 'CouponUsed' and int(user_df['CouponUsed'].values[0]) > 3:
                        suggestions.append(actionable_suggestions[feature])
                    else:
                        suggestions.append(actionable_suggestions[feature])

        # Create result dictionary to pass to template
        result = {
            'churn_status': churn_status,
            'churn_probability': f"{churn_probability:.2f}%",
            'input_data': user_input_data,
            'suggestions': suggestions
        }

    return render_template('index.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
