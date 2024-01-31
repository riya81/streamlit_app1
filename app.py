import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import hashlib
from cryptography.fernet import Fernet
import warnings

# Set page configurations
st.set_page_config(
    page_title="Election Outcome Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# User credentials (replace with your own)
valid_username = "user123"
valid_password_hash = hashlib.sha256("password123".encode()).hexdigest()

# Login Page
def login_page():
    st.title("POLLPREDICT")
    st.subheader("LOGIN")
    # Add custom styling using Markdown and HTML/CSS
    st.markdown(
        """
        <style>
            div.stTitle {
                font-size: 36px !important;
                text-align: center !important;
            }
            div.stMarkdown {
                font-size: 18px !important;
                text-align: center !important;
            }
            div.stTextInput {
                font-size: 20px !important;
                text-align: center !important;
                width: 300px !important;  /* Adjust width as needed */
                margin: 0 auto !important; /* Center the input boxes */
            }
            div.stButton {
                font-size: 24px !important;
                margin-top: 20px !important;
                padding: 10px !important;
                width: 200px !important;
                text-align: center !important;
                margin: 0 auto !important; /* Center the button */
            }
            div.stText {
                font-size: 20px !important;
                text-align: center !important;
            }
            div.stError {
                font-size: 20px !important;
                color: red !important;
                text-align: center !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == valid_username and hashlib.sha256(password.encode()).hexdigest() == valid_password_hash:
            st.session_state.login = True
        else:
            st.error("Invalid Credentials. Please try again.")

# File upload page
def upload_file_page():
    st.title("POLLPREDICT")
    st.subheader("UPLOAD")
    
    # Upload file through Streamlit
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        voting_df = pd.read_csv(uploaded_file)
        st.write(voting_df)

        # Encrypt and save the dataset
        key = Fernet.generate_key()
        f = Fernet(key)
        serialized_data = voting_df.to_json()
        encrypted_data = f.encrypt(serialized_data.encode('utf-8'))

        st.session_state.file_key = key
        st.session_state.encrypted_data = encrypted_data

        # Store voting_df in session state
        st.session_state.voting_df = voting_df

        st.success("File uploaded and encrypted successfully!")
        st.sidebar.success("File uploaded and encrypted successfully!")
    else:
        st.warning("Please upload a file to proceed.")

# Prediction Page
def prediction_page():
    st.title("POLLPREDICT")
    st.subheader("Prediction")

    # Check if logged in
    if not st.session_state.login:
        st.warning("Please login first.")
        return

    # Check if file is uploaded
    if "file_key" not in st.session_state:
        st.warning("Please upload a file first.")
        return

    st.markdown(
        """
        <style>
            /* Add your custom CSS styles here */
            div.input-container {
                margin-bottom: 15px;
            }
            div.hashed-info {
                font-style: italic;
                margin-top: 10px;
                color: #555;
            }
            /* Reduce the width of input boxes */
            div.stNumberInput{
                font-size: 20px !important;
                text-align: center !important;
                width: 300px !important;  /* Adjust width as needed */
            }
            
        </style>
        """,
        unsafe_allow_html=True
    )
    evm_votes = st.number_input("EVM Votes", min_value=0)
    postal_votes = st.number_input("Postal Votes", min_value=0)
    total_votes = st.number_input("Total Votes", min_value=0)
    percent_of_votes = st.number_input("Percentage of Votes", min_value=0.0, max_value=1.0, step=0.01)
    constituency_votes_polled = st.number_input("Constituency Votes Polled", min_value=0)
    total_votes_by_parties = st.number_input("Total Votes by Parties", min_value=0)

    key = st.session_state.file_key
    encrypted_data = st.session_state.encrypted_data

    f = Fernet(key)

    # Decrypt the encrypted dataset
    decrypted_data = f.decrypt(encrypted_data).decode('utf-8')

    # Convert the decrypted string back to a DataFrame
    decrypted_df = pd.read_json(decrypted_data)

    # Drop rows with missing values
    df = decrypted_df.dropna()

    # Drop non-numeric columns (adjust as needed)
    df = decrypted_df.drop(columns=['Constituency', 'Candidate', 'Party', 'Winning_votes'])

    # Split data into features (X) and target variable (y)
    X = decrypted_df.drop('Win_Lost_Flag', axis=1)
    y = decrypted_df['Win_Lost_Flag']

    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Ensure that the feature names match the ones used during training
    missing_features = set(X.columns) - set(df.columns)
    if missing_features:
        # Add missing columns with zeros
        zeros_df = pd.DataFrame(0, index=df.index, columns=list(missing_features))
        df = pd.concat([df, zeros_df], axis=1)

    # Reorder columns to match the original order during training
    df = df[X.columns]

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Create a sample data point for prediction
    sample_data = pd.DataFrame({
        'EVM_Votes': [evm_votes],
        'Postal_Votes': [postal_votes],
        'Total_Votes': [total_votes],
        '%_of_Votes': [percent_of_votes],
        'Tot_Constituency_votes_polled': [constituency_votes_polled],
        'Tot_votes_by_parties': [total_votes_by_parties]
    })

    # Ensure that the feature names match the ones used during training
    missing_features = set(X.columns) - set(sample_data.columns)
    if missing_features:
        # Add missing columns with zeros
        zeros_df = pd.DataFrame(0, index=sample_data.index, columns=list(missing_features))
        sample_data = pd.concat([sample_data, zeros_df], axis=1)

    # Reorder columns to match the original order during training
    sample_data = sample_data[X.columns]

    # Predict function
    def predict():
        prediction = model.predict(sample_data)
        return prediction[0]

    # Predict button
    if st.button("Predict"):
        result = predict()
        st.success(f"Predicted Win/Loss Flag: {result}")

def visualization_page():
    st.title("Visualization Page")

    # Check if logged in
    if not st.session_state.login:
        st.warning("Please login first.")
        return

    # Check if file is uploaded
    if "file_key" not in st.session_state:
        st.warning("Please upload a file first.")
        return

    # Access voting_df from session_state
    voting_df = st.session_state.voting_df

    # Visualization code...
    plt.figure(figsize=(8, 6))
    plt.tight_layout()

    if voting_df is not None:
        datar = voting_df.groupby('Party')['Win_Lost_Flag'].sum('Win_Lost_Flag')
        datar = datar[datar.values > 0]

        # Customize plot background color
        plt.rcParams['axes.facecolor'] = '#c4c3d0'  # Change to your desired background color

        # Customize bar colors
        colors = ['#800080', '#9370db', '#966fd6', '#b19cd9','#dcd0ff','#d8bfd8','#bf94e4','#d473d4']  # Replace with your desired colors
        ax = datar.plot(kind='bar', color=colors)
        ax.bar_label(ax.containers[0])

        # Customize other plot properties if needed
        ax.set_xlabel('Party')
        ax.set_ylabel('Win/Loss')
        ax.set_title("Win/Loss Distribution by Party")

        # Display the plot using st.pyplot()
        st.pyplot(plt.gcf()) 
    else:
        st.warning("No data available for visualization.")


def main():
    if "login" not in st.session_state:
        st.session_state.login = False

    if not st.session_state.login:
        login_page()
    else:
        page = st.sidebar.selectbox("Select a page", ["Upload File", "Prediction", "Visualization"])

        if page == "Upload File":
            upload_file_page()
        elif page == "Prediction":
            prediction_page()
        elif page == "Visualization":
            visualization_page()

if __name__ == '__main__':
    main()
