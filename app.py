from flask import Flask, render_template, request
import pickle
import numpy as np
from models import db, PredictionHistory

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///student_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Load the trained ML model
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

# Mappings for categorical values
motivation_map = {"Low": 0, "Medium": 1, "High": 2}
parent_support_map = {"Low": 0, "High": 1}

# ----- Utility functions -----
def calculate_ssi(data):
    """Calculate Student Success Index based on weighted factors."""
    weights = {
        'attendance': 0.25,
        'study_hours': 0.20,
        'prev_grade': 0.30,
        'motivation': 0.15,
        'parent_support': 0.10
    }
    score = (
        data['attendance'] * weights['attendance'] +
        data['study_hours'] * weights['study_hours'] +
        data['prev_grade'] * weights['prev_grade'] +
        motivation_map.get(data['motivation'], 0.5) * weights['motivation'] * 100 +
        parent_support_map.get(data['parent_support'], 0.5) * weights['parent_support'] * 100
    )

    if score >= 85:
        tier = 'High Success'
    elif score >= 70:
        tier = 'Medium Success'
    elif score >= 55:
        tier = 'Standard'
    else:
        tier = 'At Risk'

    return round(score, 2), tier

def generate_recommendations(tier):
    """Return personalized recommendations based on the performance tier."""
    recs = {
        'High Success': [
            "Keep up the excellent work.",
            "Explore leadership and advanced courses.",
            "Maintain consistent study and motivation."
        ],
        'Medium Success': [
            "Focus on improving weaker subjects.",
            "Increase attendance and study hours.",
            "Seek extra help or tutoring if needed."
        ],
        'Standard': [
            "Set achievable study goals.",
            "Improve motivation with rewards.",
            "Attend counseling sessions for support."
        ],
        'At Risk': [
            "Immediate intervention required.",
            "Increase study time significantly.",
            "Consult teachers and parents for an action plan."
        ]
    }
    return recs.get(tier, [])

# Create DB tables before first request
with app.app_context():
    db.create_all()

# ----- Routes -----
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Collect form data
        data = {
            "study_hours": float(request.form["study_hours"]),
            "attendance": float(request.form["attendance"]),
            "prev_grade": float(request.form["prev_grade"]),
            "motivation": request.form["motivation"],
            "parent_support": request.form["parent_support"]
        }

        # Prepare features for ML model
        features = np.array([[
            data["study_hours"],
            data["attendance"],
            data["prev_grade"],
            motivation_map.get(data["motivation"], 1),
            parent_support_map.get(data["parent_support"], 1)
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction] * 100

        # Calculate SSI and tier
        ssi_score, risk_tier = calculate_ssi(data)
        recommendations = generate_recommendations(risk_tier)

        # Save to database
        record = PredictionHistory(
            input_data=data,
            ssi_score=ssi_score,
            risk_tier=risk_tier,
            recommendations=recommendations
        )
        db.session.add(record)
        db.session.commit()

        # Render results page
        return render_template("results.html",
                               ssi_score=ssi_score,
                               risk_tier=risk_tier,
                               confidence=round(confidence, 2),
                               recommendations=recommendations)

    # Default GET route
    return render_template("index.html")

@app.route("/analytics")
def analytics():
    # Fetch all prediction records
    history = PredictionHistory.query.all()

    # Calculate tier counts
    tier_counts = {"High Success": 0, "Medium Success": 0, "Standard": 0, "At Risk": 0}
    ssi_scores = []

    for rec in history:
        tier_counts[rec.risk_tier] = tier_counts.get(rec.risk_tier, 0) + 1
        ssi_scores.append(rec.ssi_score)

    avg_ssi = round(sum(ssi_scores) / len(ssi_scores), 2) if ssi_scores else 0

    return render_template("analytics.html",
                           performance_data=tier_counts,
                           ssi_data=avg_ssi)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
