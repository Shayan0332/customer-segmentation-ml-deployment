from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model files
kmeans = joblib.load("kmeans_customers_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get values from form
        orders = float(request.form["orders"])
        quantity = float(request.form["quantity"])
        spending = float(request.form["spending"])

        # Create DataFrame with SAME feature names
        sample = pd.DataFrame(
            [[orders, quantity, spending]],
            columns=features
        )

        # Scale input
        sample_scaled = scaler.transform(sample)

        # Predict cluster
        cluster = kmeans.predict(sample_scaled)[0]

        # Friendly explanation
        cluster_map = {
            0: "Low Value Customer",
            1: "Regular Customer",
            2: "High Spending Customer",
            3: "VIP Customer"
        }

        prediction = cluster_map.get(cluster, "Unknown")

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
