from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get form data
        input_data = {
            "account_length": int(request.form["account_length"]),
            "area_code": int(request.form["area_code"]),
            "international_plan": request.form["international_plan"],
            "voice_mail_plan": request.form["voice_mail_plan"],
            "number_vmail_messages": int(request.form["number_vmail_messages"]),
            "total_day_minutes": float(request.form["total_day_minutes"]),
            "total_day_calls": int(request.form["total_day_calls"]),
            "total_day_charge": float(request.form["total_day_charge"]),
            "total_eve_minutes": float(request.form["total_eve_minutes"]),
            "total_eve_calls": int(request.form["total_eve_calls"]),
            "total_eve_charge": float(request.form["total_eve_charge"]),
            "total_night_minutes": float(request.form["total_night_minutes"]),
            "total_night_calls": int(request.form["total_night_calls"]),
            "total_night_charge": float(request.form["total_night_charge"]),
            "total_intl_minutes": float(request.form["total_intl_minutes"]),
            "total_intl_calls": int(request.form["total_intl_calls"]),
            "total_intl_charge": float(request.form["total_intl_charge"]),
            "customer_service_calls": int(request.form["customer_service_calls"]),
            "churn": False  # Not used for prediction
        }

        # Send request to FastAPI
        response = requests.post(FASTAPI_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json()["prediction"][0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
