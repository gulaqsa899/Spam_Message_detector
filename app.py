import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("📱 Spam Message Detector")

st.write("Enter a message to check whether it is spam or normal.")

# Message input
message = st.text_area("Enter your message here:")

# Check button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Convert message into numbers
        message_vector = vectorizer.transform([message])

        # Predict message category
        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM!")

        else:
            st.success("✅ This message is NORMAL (HAM).")