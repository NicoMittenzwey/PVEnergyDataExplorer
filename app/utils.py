# app/utils.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_season(date):
    """
    Determine the season for a given date.
    """
    month = date.month
    day = date.day

    winter_mid = datetime(date.year, 12, 21).date()
    spring_mid = datetime(date.year, 3, 21).date()
    summer_mid = datetime(date.year, 6, 21).date()
    autumn_mid = datetime(date.year, 9, 21).date()

    if date.month < 3:
        winter_mid = datetime(date.year - 1, 12, 21).date()

    date_ordinal = date.toordinal()

    distances = [
        (abs(date_ordinal - winter_mid.toordinal()), 'Winter'),
        (abs(date_ordinal - spring_mid.toordinal()), 'Spring'),
        (abs(date_ordinal - summer_mid.toordinal()), 'Summer'),
        (abs(date_ordinal - autumn_mid.toordinal()), 'Autumn')
    ]

    return min(distances, key=lambda x: x[0])[1]

def process_data(df, is_generation=False, scaling_factor=1.0):
    """
    Process the uploaded data and prepare it for analysis.
    """
    # Check if required columns are present
    required_columns = {'Date', 'Time', 'kW'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file is missing required columns: {required_columns}")

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['kW'] = pd.to_numeric(df['kW'], errors='coerce')
    if is_generation:
        df['kW'] *= scaling_factor  # Apply scaling to generation data
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['season'] = df['datetime'].dt.date.apply(get_season)
    return df

def identify_potential_holidays(df):
    """
    Identify potential holidays based on consumption patterns.
    """
    daily_consumption = df.groupby(df['datetime'].dt.date)['kW'].sum().reset_index()
    daily_consumption['datetime'] = pd.to_datetime(daily_consumption['datetime'])
    daily_consumption['is_weekend'] = daily_consumption['datetime'].dt.dayofweek.isin([5, 6])

    weekday_consumption = daily_consumption[~daily_consumption['is_weekend']]['kW']
    weekend_consumption = daily_consumption[daily_consumption['is_weekend']]['kW']

    weekday_mean = weekday_consumption.mean()
    weekday_std = weekday_consumption.std()
    daily_consumption['z_score'] = (daily_consumption['kW'] - weekday_mean) / weekday_std

    weekend_threshold = weekend_consumption.mean()
    potential_holidays = daily_consumption[
        (~daily_consumption['is_weekend']) &
        (daily_consumption['kW'] < weekend_threshold) &
        (daily_consumption['z_score'] < -1.5)
    ]['datetime'].dt.date.tolist()

    return potential_holidays

#Option 2: If you don't need to convert to datetime objects, it's more efficient to work directly with the datetime.time objects and extract the needed information.

def get_day_data(df, date=None, is_weekend=None, potential_holidays=[], is_generation=False):
    if date is not None:
        day_data = df[df['datetime'].dt.date == date].copy()
    elif is_weekend is not None:
        if not is_weekend and not is_generation:
            df = df[~df['datetime'].dt.date.isin(potential_holidays)]
        day_data = df[df['is_weekend'] == is_weekend].groupby(df['datetime'].dt.time).agg({'kW': 'mean'}).reset_index()
    else:
        day_data = df.groupby(df['datetime'].dt.time).agg({'kW': 'mean'}).reset_index()

    # Directly compute 'minutes' from 'datetime' without conversion
    day_data['minutes'] = day_data['datetime'].apply(lambda t: t.hour * 60 + t.minute)
    return day_data


def simulate_battery(df_consumption, df_generation, battery_capacity):
    """
    Simulate battery usage based on consumption and generation data.
    """
    # Ensure datetime columns are in the correct format
    df_consumption['datetime'] = pd.to_datetime(df_consumption['datetime'])
    df_generation['datetime'] = pd.to_datetime(df_generation['datetime'])

    # Round timestamps to nearest 15-minute interval
    df_consumption['datetime_rounded'] = df_consumption['datetime'].dt.round('15min')
    df_generation['datetime_rounded'] = df_generation['datetime'].dt.round('15min')

    # Merge consumption and generation data
    merged_data = pd.merge(df_consumption, df_generation, on='datetime_rounded', suffixes=('_consumption', '_generation'))

    if merged_data.empty:
        return pd.DataFrame()

    merged_data = merged_data.sort_values('datetime_rounded')

    # Initialize battery state and energy flow variables
    battery_charge = 0  # Start with 0 kWh
    battery_charge_history = []
    energy_from_battery = []
    energy_to_grid = []
    energy_from_grid = []

    for _, row in merged_data.iterrows():
        # Convert kW to kWh for 15-minute intervals
        consumption = row['kW_consumption'] / 4  # kWh for 15 minutes
        generation = row['kW_generation'] / 4  # kWh for 15 minutes
        net_energy = generation - consumption

        if net_energy > 0:  # Excess generation
            energy_to_battery = min(net_energy, battery_capacity - battery_charge)
            battery_charge += energy_to_battery
            energy_to_grid.append(net_energy - energy_to_battery)
            energy_from_battery.append(0)
            energy_from_grid.append(0)
        else:  # Energy deficit
            energy_from_battery_needed = min(abs(net_energy), battery_charge)
            battery_charge -= energy_from_battery_needed
            energy_from_grid.append(abs(net_energy) - energy_from_battery_needed)
            energy_from_battery.append(energy_from_battery_needed)
            energy_to_grid.append(0)

        battery_charge_history.append(battery_charge)

    merged_data['battery_charge'] = battery_charge_history
    merged_data['energy_from_battery'] = energy_from_battery
    merged_data['energy_to_grid'] = energy_to_grid
    merged_data['energy_from_grid'] = energy_from_grid

    return merged_data

def plot_to_img(fig):
    """
    Convert a Matplotlib figure to a base64-encoded image.
    """
    pngImage = io.BytesIO()
    fig.savefig(pngImage, format='png', bbox_inches='tight')
    pngImage.seek(0)
    png_base64 = base64.b64encode(pngImage.read()).decode('ascii')
    plt.close(fig)
    return png_base64

def run_analysis(df_consumption, df_generation, potential_holidays, scaling_factor_value, battery_capacity_value):
    """
    Run the entire analysis and return results.
    """
    original_analysis_fig = plot_original_analysis(
        df_consumption, df_generation, potential_holidays, scaling_factor_value
    )

    energy_analysis_results = analyze_energy(
        df_consumption, df_generation, scaling_factor_value
    )

    battery_simulation_results = run_battery_simulation(
        df_consumption, df_generation, battery_capacity_value
    )

    return {
        'original_analysis_fig': original_analysis_fig,
        'energy_analysis_results': energy_analysis_results,
        'battery_simulation_results': battery_simulation_results,
        'scaling_factor_value': scaling_factor_value,
        'battery_capacity_value': battery_capacity_value
    }

def plot_original_analysis(df_consumption, df_generation, potential_holidays, scaling_factor_value):
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.ravel()

    for idx, season in enumerate(seasons):
        season_consumption = df_consumption[df_consumption['season'] == season]
        season_generation = df_generation[df_generation['season'] == season]

        if season_consumption.empty or season_generation.empty:
            axs[idx].text(0.5, 0.5, f"No data available for {season}",
                          ha='center', va='center', transform=axs[idx].transAxes)
            continue

        # Calculate daily total consumption
        season_daily_consumption = season_consumption.groupby(season_consumption['datetime'].dt.date)['kW'].sum().reset_index()
        season_daily_consumption.rename(columns={'datetime': 'date'}, inplace=True)

        # Ensure 'date' column is datetime.date objects
        # season_daily_consumption['date'] is already datetime.date from .dt.date

        # Find max weekday consumption day for the season
        mask = (
            (season_daily_consumption['kW'] == season_daily_consumption[~season_daily_consumption['date'].isin(potential_holidays)]['kW'].max()) &
            (~season_daily_consumption['date'].apply(lambda x: x.weekday()).isin([5, 6])) &
            (~season_daily_consumption['date'].isin(potential_holidays))
        )
        if mask.any():
            max_weekday_date = season_daily_consumption.loc[mask, 'date'].iloc[0]
        else:
            max_weekday_date = season_daily_consumption['date'].iloc[0]  # Fallback

        # Find min consumption day for the season
        min_consumption_date = season_daily_consumption.loc[season_daily_consumption['kW'].idxmin(), 'date']

        # Get data for plotting
        avg_generation_data = get_day_data(season_generation, is_generation=True)
        avg_weekday_data = get_day_data(season_consumption, is_weekend=False, potential_holidays=potential_holidays)
        max_weekday_data = get_day_data(season_consumption, date=max_weekday_date)
        avg_weekend_data = get_day_data(season_consumption, is_weekend=True)
        min_consumption_data = get_day_data(season_consumption, date=min_consumption_date)

        # Plot data
        axs[idx].plot(avg_generation_data['minutes'], avg_generation_data['kW'],
                      label=f'Avg Generation (scaled by {scaling_factor_value:.2f})', color='green')
        axs[idx].plot(avg_weekday_data['minutes'], avg_weekday_data['kW'], label='Avg Weekday', color='blue')
        axs[idx].plot(max_weekday_data['minutes'], max_weekday_data['kW'], label=f'Max Weekday ({max_weekday_date})', color='red')
        axs[idx].plot(avg_weekend_data['minutes'], avg_weekend_data['kW'], label='Avg Weekend', color='purple')
        axs[idx].plot(min_consumption_data['minutes'], min_consumption_data['kW'], label=f'Min Consumption ({min_consumption_date})', color='orange')

        axs[idx].set_title(f'{season} Power Consumption and Generation')
        axs[idx].set_xlabel('Time')
        axs[idx].set_ylabel('Power (kW)')
        axs[idx].grid(True)

        # Format x-axis to show time
        def minutes_to_hhmm(x, pos):
            return f'{int(x // 60):02d}:{int(x % 60):02d}'

        axs[idx].xaxis.set_major_formatter(plt.FuncFormatter(minutes_to_hhmm))
        axs[idx].xaxis.set_major_locator(plt.MultipleLocator(120))  # Tick every 2 hours

        # Add legend
        axs[idx].legend()

    plt.tight_layout()
    fig_data = plot_to_img(fig)
    return fig_data


def analyze_energy(df_consumption, df_generation, scaling_factor_value):
    """
    Analyze energy usage and generation.
    """
    # Calculate daily energy totals
    daily_energy = pd.merge(
        df_consumption.groupby(df_consumption['datetime'].dt.date)['kW'].sum().reset_index(name='consumer_usage_kWh'),
        df_generation.groupby(df_generation['datetime'].dt.date)['kW'].sum().reset_index(name='generation_kWh'),
        on='datetime'
    )

    # Calculate unused generated energy and grid consumption
    daily_energy['unused_generated_kWh'] = (daily_energy['generation_kWh'] - daily_energy['consumer_usage_kWh']).clip(lower=0)
    daily_energy['grid_consumption_kWh'] = (daily_energy['consumer_usage_kWh'] - daily_energy['generation_kWh']).clip(lower=0)

    # Add season information
    daily_energy['season'] = daily_energy['datetime'].apply(get_season)

    # Calculate total energy metrics
    total_generation = daily_energy['generation_kWh'].sum()
    total_consumer_usage = daily_energy['consumer_usage_kWh'].sum()
    total_unused_generated = daily_energy['unused_generated_kWh'].sum()
    total_grid_consumption = daily_energy['grid_consumption_kWh'].sum()

    coverage_percentage = (total_generation / total_consumer_usage * 100) if total_consumer_usage > 0 else 0

    overall_results = {
        'total_generation': total_generation,
        'total_consumer_usage': total_consumer_usage,
        'total_unused_generated': total_unused_generated,
        'total_grid_consumption': total_grid_consumption,
        'coverage_percentage': coverage_percentage
    }

    # Analyze by season
    seasonal_energy = daily_energy.groupby('season').agg({
        'generation_kWh': 'sum',
        'consumer_usage_kWh': 'sum',
        'unused_generated_kWh': 'sum',
        'grid_consumption_kWh': 'sum'
    }).reset_index()

    seasonal_energy['generation_coverage_percentage'] = (
        seasonal_energy['generation_kWh'] / seasonal_energy['consumer_usage_kWh'] * 100
    ).clip(upper=100)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.2
    index = np.arange(len(seasonal_energy))

    ax.bar(index, seasonal_energy['generation_kWh'], bar_width, label='Generated', color='green')
    ax.bar(index + bar_width, seasonal_energy['consumer_usage_kWh'], bar_width, label='Consumer Usage', color='blue')
    ax.bar(index + 2 * bar_width, seasonal_energy['unused_generated_kWh'], bar_width, label='Unused Generated', color='yellow')
    ax.bar(index + 3 * bar_width, seasonal_energy['grid_consumption_kWh'], bar_width, label='Grid Consumption', color='red')

    ax.set_xlabel('Season')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title(f'Seasonal Energy Analysis (Generation Scaling Factor: {scaling_factor_value:.2f})')
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(seasonal_energy['season'])
    ax.legend()

    plt.tight_layout()
    fig_data = plot_to_img(fig)

    return {
        'overall_results': overall_results,
        'seasonal_energy': seasonal_energy.to_dict(orient='records'),
        'energy_analysis_fig': fig_data
    }

def run_battery_simulation(df_consumption, df_generation, battery_capacity_value):
    """
    Run the battery simulation and generate results.
    """
    simulation_data = simulate_battery(df_consumption, df_generation, battery_capacity_value)

    if simulation_data.empty:
        return {
            'error': 'No data available for the battery simulation.'
        }

    # Plot battery simulation results
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot consumption, generation, and battery charge
    ax.plot(simulation_data['datetime_rounded'], simulation_data['kW_consumption'], label='Consumption', color='red')
    ax.plot(simulation_data['datetime_rounded'], simulation_data['kW_generation'], label='Generation', color='green')
    ax.plot(simulation_data['datetime_rounded'], simulation_data['battery_charge'], label='Battery Charge', color='blue')

    ax.set_ylabel('Power (kW) / Battery Charge (kWh)')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_title(f'Battery Simulation (Capacity: {battery_capacity_value} kWh)')

    # Set x-axis to show dates and times
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig_data = plot_to_img(fig)

    # Calculate energy flow statistics
    total_consumption = simulation_data['kW_consumption'].sum() / 4  # Convert to kWh
    total_generation = simulation_data['kW_generation'].sum() / 4  # Convert to kWh
    energy_from_battery = sum(simulation_data['energy_from_battery'])
    energy_to_grid = sum(simulation_data['energy_to_grid'])
    energy_from_grid = sum(simulation_data['energy_from_grid'])
    final_charge = simulation_data['battery_charge'].iloc[-1]

    # Calculate self-consumption and autarky rates
    total_self_consumption = total_generation - energy_to_grid
    self_consumption_rate = (total_self_consumption / total_generation * 100) if total_generation > 0 else 0
    autarky_rate = ((total_consumption - energy_from_grid) / total_consumption * 100) if total_consumption > 0 else 0

    battery_utilization = (final_charge / battery_capacity_value * 100) if battery_capacity_value > 0 else 0

    simulation_results = {
        'total_consumption': total_consumption,
        'total_generation': total_generation,
        'energy_from_battery': energy_from_battery,
        'energy_to_grid': energy_to_grid,
        'energy_from_grid': energy_from_grid,
        'final_charge': final_charge,
        'battery_utilization': battery_utilization,
        'self_consumption_rate': self_consumption_rate,
        'autarky_rate': autarky_rate,
        'battery_simulation_fig': fig_data
    }

    return simulation_results

