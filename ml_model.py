import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Generate synthetic student dataset
def generate_data(n=500):
    np.random.seed(42)
    study_hours = np.random.uniform(0, 30, n)
    attendance = np.random.uniform(50, 100, n)
    prev_grade = np.random.uniform(40, 100, n)
    motivation = np.random.choice([0, 1, 2], n)
    parent_support = np.random.choice([0, 1], n)  # 0=Low,1=High

    score = (
        study_hours * 0.25 +
        attendance * 0.25 +
        prev_grade * 0.30 +
        motivation * 10 +
        parent_support * 10
    )

    performance = (score > 100).astype(int)  # 1=Good,0=At Risk

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "prev_grade": prev_grade,
        "motivation": motivation,
        "parent_support": parent_support,
        "performance": performance
    })
    return df

def train_model():
    df = generate_data()
    X = df[["study_hours", "attendance", "prev_grade", "motivation", "parent_support"]]
    y = df["performance"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open("student_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained. Accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    train_model()
