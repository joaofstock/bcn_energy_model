import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load dictionaries
with open('codi_avg_energy_dict.pkl', 'rb') as file:
    codi_avg_energy_dict = pickle.load(file)

with open('codi_avg_energyperperson_dict.pkl', 'rb') as file:
    codi_avg_energyperperson_dict = pickle.load(file)

with open('daytime_month_avg_energy_dict.pkl', 'rb') as file:
    daytime_month_avg_energy_dict = pickle.load(file)

# Load assembled_dfs for historical data
with open("assembled_dfs.pkl", "rb") as f:
    assembled_dfs = pickle.load(f)

# Load the synthetic dataset as DataFrame
try:
    with open('synthetic_data.pkl', 'rb') as file:
        synt_data = pickle.load(file)  # This should be a DataFrame
        if not isinstance(synt_data, pd.DataFrame):
            raise ValueError("Synthetic data is not a DataFrame")
except Exception as e:
    print(f"Error loading the synthetic dataset: {e}")
    synt_data = pd.DataFrame()  # Default to an empty DataFrame

# Load the machine learning model
try:
    with open('pickled_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Load neighborhood mapping dictionary
with open('HOOD_MAPPING.pkl', 'rb') as file:
    HOOD_MAPPING = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

def validate_input(neighborhood_name, start_year, start_month, end_year, end_month):
    if neighborhood_name not in HOOD_MAPPING:
        return f"Neighborhood '{neighborhood_name}' is not valid."

    if start_year == end_year and start_month > end_month:
        return "Start month cannot be greater than end month within the same year."

    if start_year > end_year:
        return "Start year cannot be greater than end year."

    if (start_year < 2024 or (start_year == 2024 and start_month < 11)):
        return "Start year-month must be 2024-11 or later."

    if (end_year > 2026 or (end_year == 2026 and end_month > 12)):
        return "End year-month must be 2026-12 or earlier."

    return None

def adjust_data(synthetic_data, population_change, temperature_change):
    try:
        # Apply percentage changes to population and temperature
        synthetic_data['population'] *= (1 + population_change / 100)
        synthetic_data['temp'] *= (1 + temperature_change / 100)
        print("Synthetic data adjusted for population and temperature changes.")
    except Exception as e:
        print(f"Error adjusting synthetic data: {e}")
        raise


def prepare_and_predict(neighborhood_name, start_year, start_month, end_year, end_month, population_change, temperature_change):
    if model is None:
        return None, "Model is not loaded properly."

    # Adjust synthetic data based on user input percentages
    adjust_data(synt_data, population_change, temperature_change)

    # Map neighborhood name to `Codi_barri`
    codi_barri = HOOD_MAPPING.get(neighborhood_name)
    if not codi_barri:
        return None, f"Neighborhood '{neighborhood_name}' is not valid."

    # Filter synthetic data for the given neighborhood and date range
    try:
        filtered_data = synt_data[
            (synt_data['Codi_barri'] == codi_barri) &
            (
                (synt_data['year'] > start_year) |
                ((synt_data['year'] == start_year) & (synt_data['month'] >= start_month))
            ) &
            (
                (synt_data['year'] < end_year) |
                ((synt_data['year'] == end_year) & (synt_data['month'] <= end_month))
            )
        ].copy()  # Explicitly create a copy to avoid SettingWithCopyWarning
    except Exception as e:
        print(f"Error filtering data: {e}")
        return None, "Error filtering synthetic data."

    if filtered_data.empty:
        return None, "No data found for the selected criteria."

    print("Filtered Data Preview:")
    print(filtered_data.head())

    # Retain relevant metadata for output
    metadata = filtered_data[['year', 'month', 'daytime']].reset_index(drop=True)

    # Replace or compute required features using dictionaries
    try:
        filtered_data['daytime_month_avg_energy'] = filtered_data.apply(
            lambda row: daytime_month_avg_energy_dict.get((row['daytime'], row['month']), 0),
            axis=1
        )

        filtered_data['energy/person'] = filtered_data['Codi_barri'].replace(codi_avg_energyperperson_dict)
        filtered_data['codi_avg_energy'] = filtered_data['Codi_barri'].replace(codi_avg_energy_dict)
    except KeyError as e:
        print(f"Key error during feature replacement: {e}")
        return None, f"Key error in dictionary lookup: {str(e)}"

    print("Filtered Data After Feature Replacement:")
    print(filtered_data.head())

    # Select only features required for the model
    try:
        model_features = filtered_data[
            ['temp', 'area_m2', 'age', 'population', 'codi_avg_energy', 'energy/person', 'daytime_month_avg_energy']
        ]
    except KeyError as e:
        print(f"Missing model feature columns: {e}")
        return None, "Missing required model features."

    print("Model Features Preview:")
    print(model_features.head())

    # Make predictions
    try:
        predictions = model.predict(model_features)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, f"Prediction error: {str(e)}"

    # Add predictions to metadata
    metadata['energy_amount'] = predictions

    # Drop the 'Codi_barri' column from metadata if present
    if 'Codi_barri' in metadata.columns:
        metadata = metadata.drop(columns=['Codi_barri'])

    # Aggregate results by year and month
    aggregated_results = metadata.groupby(['year', 'month'], as_index=False).agg(
        avg_energy_amount=('energy_amount', 'mean'),
        total_energy_amount=('energy_amount', 'sum')
    )

    return {
        'aggregated_results': aggregated_results,
        'details': metadata
    }, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate input data
        data = request.json
        neighborhood_name = data.get('hood_name')
        start_year = int(data.get('start_year'))
        start_month = int(data.get('start_month'))
        end_year = int(data.get('end_year'))
        end_month = int(data.get('end_month'))
        population_change = float(data.get('population_change', 0))  # Default to 0 if not provided
        temperature_change = float(data.get('temperature_change', 0))  # Default to 0 if not provided

        validation_error = validate_input(neighborhood_name, start_year, start_month, end_year, end_month)
        if validation_error:
            return jsonify({'error': validation_error}), 400

        # Prepare data and make predictions
        result, error = prepare_and_predict(
            neighborhood_name, start_year, start_month, end_year, end_month,
            population_change, temperature_change
        )
        if error:
            return jsonify({'error': error}), 404

        return jsonify({
            'aggregated_results': result['aggregated_results'].to_dict(orient='records'),
            'details': result['details'].to_dict(orient='records')
        })

    except Exception as e:
        # Log detailed error
        error_details = traceback.format_exc()
        print(f"Error occurred: {error_details}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

