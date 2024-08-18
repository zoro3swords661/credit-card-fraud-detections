```markdown
# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using a machine learning model. The application is built using Python and Streamlit for the web interface.

## Features

- **Data Loading**: Load credit card transaction data from a CSV file.
- **Data Preprocessing**: Scale features and split the data into training and testing sets.
- **Model Training**: Train a `RandomForestClassifier` to detect fraudulent transactions.
- **Prediction**: Predict whether a transaction is fraudulent based on user input.
- **Visualizations**: Display various visualizations such as transaction distribution, correlation matrix, and class distribution.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thisisharshsah/credit-card-fraud-detections.git
   cd credit-card-fraud-detections
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Adding Data

1. Download the `creditcard.csv` file from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).
2. Place the `creditcard.csv` file in the root directory of the project.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

## How to Use

1. **Upload Data**: Upload a CSV file containing credit card transaction data.
2. **Train Model**: Click the "Train Model" button to train the model.
3. **Input Features**: Enter the feature values for a transaction.
4. **Predict**: Click the "Predict" button to determine if the transaction is fraudulent.
5. **Visualizations**: Use the buttons to toggle between different visualizations.

## File Structure

- `streamlit_app.py`: Main application file.
- `requirements.txt`: List of required Python packages.

## Author

This project is developed by Dinesh Lal.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```

Now the numbering is correct. Let me know if you need further adjustments!