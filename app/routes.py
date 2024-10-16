# app/routes.py
from flask import render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import pandas as pd
import os
import warnings

from app import app
from app.utils import (
    allowed_file, process_data, identify_potential_holidays, run_analysis
)

# Suppress warnings
warnings.filterwarnings('ignore')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    # Check if the file has an allowed extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Validate and sanitize user inputs
            scaling_factor_value = float(request.form.get('scaling_factor', 1.0))
            if not (0.1 <= scaling_factor_value <= 5.0):
                return "Scaling factor must be between 0.1 and 5.0", 400

            battery_capacity_value = float(request.form.get('battery_capacity', 10))
            if not (0 <= battery_capacity_value <= 100):
                return "Battery capacity must be between 0 and 100 kWh", 400

            # Check if the files are present
            if 'consumption_file' not in request.files or 'generation_file' not in request.files:
                return "Missing files", 400

            consumption_file = request.files['consumption_file']
            generation_file = request.files['generation_file']

            if consumption_file.filename == '' or generation_file.filename == '':
                return "No selected files", 400

            # Check if files have allowed extensions
            if not (allowed_file(consumption_file.filename) and allowed_file(generation_file.filename)):
                return "Files must be CSV format", 400

            # Read the CSV files into DataFrames
            df_consumption = pd.read_csv(consumption_file)
            df_generation = pd.read_csv(generation_file)

            # Get start_date and end_date from form
            start_date_str = request.form.get('start_date')
            end_date_str = request.form.get('end_date')
            start_date = end_date = None

            if start_date_str:
                start_date = pd.to_datetime(start_date_str)
            if end_date_str:
                end_date = pd.to_datetime(end_date_str)

            # Process the data
            df_consumption = process_data(df_consumption)
            df_generation = process_data(df_generation, is_generation=True, scaling_factor=scaling_factor_value)

            # Filter data by date range if dates are provided
            if start_date and end_date:
                mask_consumption = (df_consumption['datetime'] >= start_date) & (df_consumption['datetime'] <= end_date)
                df_consumption = df_consumption.loc[mask_consumption]
                mask_generation = (df_generation['datetime'] >= start_date) & (df_generation['datetime'] <= end_date)
                df_generation = df_generation.loc[mask_generation]
            elif start_date:
                df_consumption = df_consumption[df_consumption['datetime'] >= start_date]
                df_generation = df_generation[df_generation['datetime'] >= start_date]
            elif end_date:
                df_consumption = df_consumption[df_consumption['datetime'] <= end_date]
                df_generation = df_generation[df_generation['datetime'] <= end_date]
            # else: do not filter

            # Identify potential holidays
            potential_holidays = identify_potential_holidays(df_consumption)

            # Run analysis and generate plots
            analysis_results = run_analysis(
                df_consumption, df_generation, potential_holidays,
                scaling_factor_value, battery_capacity_value
            )

            # Combine the consumption and generation data for preview
            preview_data = pd.merge(
                df_consumption, df_generation, on=['datetime', 'day_of_week', 'is_weekend', 'season'], how='outer', suffixes=('_consumption', '_generation')
            )

            # Select the desired columns in the specified order
            preview_data = preview_data[['datetime', 'day_of_week', 'is_weekend', 'season', 'kW_consumption', 'kW_generation']]

            # Get the first 100 lines
            preview_data = preview_data.head(100).copy()

            # Convert datetime column to string for display
            preview_data['datetime'] = preview_data['datetime'].astype(str)

            # Convert DataFrame to HTML table
            data_preview_html = preview_data.to_html(classes='table table-bordered table-sm', index=False)

            # Unpack analysis_results to pass individual variables to the template
            original_analysis_fig = analysis_results['original_analysis_fig']
            energy_analysis_results = analysis_results['energy_analysis_results']
            battery_simulation_results = analysis_results['battery_simulation_results']

            return render_template(
                'index.html',
                original_analysis_fig=original_analysis_fig,
                energy_analysis_results=energy_analysis_results,
                battery_simulation_results=battery_simulation_results,
                scaling_factor_value=scaling_factor_value,
                battery_capacity_value=battery_capacity_value,
                data_preview_html=data_preview_html,
                data_preview=True,
                start_date_str=start_date_str,
                end_date_str=end_date_str
            )

        except Exception as e:
            # Handle exceptions and provide error feedback
            print(f"Error: {e}")
            return f"An error occurred during processing: {str(e)}", 500

    # For GET requests
    return render_template('index.html', data_preview=False)
