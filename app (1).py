import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Model Accuracy Comparison App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:", df.head())

    target_column = st.selectbox("Select the Target Column", df.columns)

    # Label Encoding
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    for col in df.columns:
       if df[col].dtype == 'object':
         df[col] = le.fit_transform(df[col])
         df


    if st.button("Train Models and Compare"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models
        log_reg = LogisticRegression(max_iter=1000)
        dt = DecisionTreeClassifier()
        rf = RandomForestClassifier()

        log_reg.fit(X_train, y_train)
        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        # Predictions
        log_reg_pred = log_reg.predict(X_test)
        dt_pred = dt.predict(X_test)
        rf_pred = rf.predict(X_test)

        # Accuracy results
        accuracies = {
            'Logistic Regression': accuracy_score(y_test, log_reg_pred),
            'Decision Tree': accuracy_score(y_test, dt_pred),
            'Random Forest': accuracy_score(y_test, rf_pred)
        }

        st.subheader("Model Accuracies")
        st.write(pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"]))

        # Best model
        best_model = max(accuracies, key=accuracies.get)
        st.success(f"Best Model: {best_model} with accuracy {accuracies[best_model]:.4f}")
