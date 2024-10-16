# tests/test_utils.py
import unittest
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app import utils
import pandas as pd
import numpy as np
from datetime import datetime
import io

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.consumption_csv = io.StringIO("""Date,Time,kW
2021-01-01,00:00,1.0
2021-01-01,00:15,1.1
2021-01-01,00:30,1.2
""")

        self.generation_csv = io.StringIO("""Date,Time,kW
2021-01-01,00:00,0.5
2021-01-01,00:15,0.6
2021-01-01,00:30,0.7
""")

        self.df_consumption = pd.read_csv(self.consumption_csv)
        self.df_generation = pd.read_csv(self.generation_csv)

    def test_allowed_file(self):
        self.assertTrue(utils.allowed_file('data.csv'))
        self.assertFalse(utils.allowed_file('data.txt'))
        self.assertFalse(utils.allowed_file('data'))

    def test_get_season(self):
        date = datetime(2021, 6, 21)
        season = utils.get_season(date)
        self.assertEqual(season, 'Summer')

    def test_process_data(self):
        df_processed = utils.process_data(self.df_consumption)
        self.assertIn('datetime', df_processed.columns)
        self.assertIn('kW', df_processed.columns)
        self.assertIn('season', df_processed.columns)
        self.assertFalse(df_processed.empty)

    def test_identify_potential_holidays(self):
        df_processed = utils.process_data(self.df_consumption)
        holidays = utils.identify_potential_holidays(df_processed)
        self.assertIsInstance(holidays, list)

    def test_get_day_data(self):
        df_processed = utils.process_data(self.df_consumption)
        day_data = utils.get_day_data(df_processed, date=datetime(2021, 1, 1).date())
        self.assertFalse(day_data.empty)
        self.assertIn('minutes', day_data.columns)
        self.assertIn('kW', day_data.columns)

    def test_simulate_battery(self):
        df_consumption_processed = utils.process_data(self.df_consumption)
        df_generation_processed = utils.process_data(self.df_generation, is_generation=True)
        battery_capacity = 10
        simulation_data = utils.simulate_battery(df_consumption_processed, df_generation_processed, battery_capacity)
        self.assertFalse(simulation_data.empty)
        self.assertIn('battery_charge', simulation_data.columns)
        self.assertIn('energy_from_battery', simulation_data.columns)

    def test_plot_to_img(self):
        # Create a simple plot
        fig, ax = utils.plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        img_data = utils.plot_to_img(fig)
        self.assertIsInstance(img_data, str)
        self.assertTrue(len(img_data) > 0)

    def test_analyze_energy(self):
        df_consumption_processed = utils.process_data(self.df_consumption)
        df_generation_processed = utils.process_data(self.df_generation, is_generation=True)
        scaling_factor_value = 1.0
        energy_analysis_results = utils.analyze_energy(df_consumption_processed, df_generation_processed, scaling_factor_value)
        self.assertIn('overall_results', energy_analysis_results)
        self.assertIn('energy_analysis_fig', energy_analysis_results)
        self.assertIn('seasonal_energy', energy_analysis_results)

    def test_run_battery_simulation(self):
        df_consumption_processed = utils.process_data(self.df_consumption)
        df_generation_processed = utils.process_data(self.df_generation, is_generation=True)
        battery_capacity_value = 10
        simulation_results = utils.run_battery_simulation(df_consumption_processed, df_generation_processed, battery_capacity_value)
        self.assertIn('battery_simulation_fig', simulation_results)
        self.assertIn('total_consumption', simulation_results)
        self.assertIn('energy_from_battery', simulation_results)

    def test_run_analysis(self):
        df_consumption_processed = utils.process_data(self.df_consumption)
        df_generation_processed = utils.process_data(self.df_generation, is_generation=True)
        potential_holidays = utils.identify_potential_holidays(df_consumption_processed)
        scaling_factor_value = 1.0
        battery_capacity_value = 10
        analysis_results = utils.run_analysis(
            df_consumption_processed,
            df_generation_processed,
            potential_holidays,
            scaling_factor_value,
            battery_capacity_value
        )
        self.assertIn('original_analysis_fig', analysis_results)
        self.assertIn('energy_analysis_results', analysis_results)
        self.assertIn('battery_simulation_results', analysis_results)

if __name__ == '__main__':
    unittest.main()
