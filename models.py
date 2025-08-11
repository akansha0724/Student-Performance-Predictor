from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Create SQLAlchemy instance
db = SQLAlchemy()

class PredictionHistory(db.Model):
    """
    Stores each prediction made by the system with input data,
    calculated SSI score, tier, recommendations, and timestamp.
    """
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=True)  # Optional, if you later add a name field
    input_data = db.Column(db.JSON, nullable=False)          # Stores the raw input as JSON
    ssi_score = db.Column(db.Float, nullable=False)
    risk_tier = db.Column(db.String(50), nullable=False)
    recommendations = db.Column(db.JSON, nullable=False)
    prediction_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PredictionHistory {self.id} - {self.risk_tier}>"

