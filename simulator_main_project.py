# simulator.py
# Simulation script to test the pricing strategy defined in main_project/strategy.py
# Generates random competitor prices and records only our team's demand history

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main_project.strategy import strategy
import datetime
import argparse


# Demand pattern models
class DemandModel:
    """Base class for demand models"""
    def __init__(self, params=None):
        self.params = params or {}
    
    def get_demand(self, price, period, history=None):
        """Return demand for a given price"""
        raise NotImplementedError("Subclasses must implement get_demand")
        
    @staticmethod
    def get_models():
        return {
            'standard': StandardDemandModel,
            'seasonal': SeasonalDemandModel,
            'shock': DemandShockModel,
            'trend': TrendDemandModel,
        }


class StandardDemandModel(DemandModel):
    """Standard demand model with Poisson distribution"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.base_factor = self.params.get('base_factor', 0.5)
        
    def get_demand(self, price, period, history=None):
        lam = max(100 - price, 1) * self.base_factor
        return np.random.poisson(lam)


class SeasonalDemandModel(DemandModel):
    """Seasonal demand model with cyclical patterns"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.base_factor = self.params.get('base_factor', 0.5)
        self.amplitude = self.params.get('amplitude', 0.2)
        self.period_length = self.params.get('period_length', 20)
        
    def get_demand(self, price, period, history=None):
        # Add seasonal component with sine wave
        season_factor = 1 + self.amplitude * np.sin(2 * np.pi * period / self.period_length)
        lam = max(100 - price, 1) * self.base_factor * season_factor
        return np.random.poisson(lam)


class DemandShockModel(DemandModel):
    """Demand model with random shocks"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.base_factor = self.params.get('base_factor', 0.5)
        self.shock_probability = self.params.get('shock_probability', 0.1)
        self.shock_factor_min = self.params.get('shock_factor_min', 0.5)
        self.shock_factor_max = self.params.get('shock_factor_max', 1.5)
        
    def get_demand(self, price, period, history=None):
        # Randomly apply shocks to demand
        shock_factor = 1.0
        if np.random.random() < self.shock_probability:
            shock_factor = np.random.uniform(self.shock_factor_min, self.shock_factor_max)
            
        lam = max(100 - price, 1) * self.base_factor * shock_factor
        return np.random.poisson(lam)


class TrendDemandModel(DemandModel):
    """Demand model with long-term trend"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.base_factor = self.params.get('base_factor', 0.5)
        self.trend_factor = self.params.get('trend_factor', 0.005)
        self.trend_type = self.params.get('trend_type', 'increasing')  # 'increasing' or 'decreasing'
        
    def get_demand(self, price, period, history=None):
        # Apply trend factor based on period
        if self.trend_type == 'increasing':
            trend = 1 + self.trend_factor * period
        else:
            trend = 1 / (1 + self.trend_factor * period)
            
        lam = max(100 - price, 1) * self.base_factor * trend
        return np.random.poisson(lam)


# Competitor pricing models
class CompetitorModel:
    """Base class for competitor pricing models"""
    def __init__(self, params=None):
        self.params = params or {}
    
    def get_price(self, period, history=None, our_price=None):
        """Return competitor price for current period"""
        raise NotImplementedError("Subclasses must implement get_price")
    
    @staticmethod
    def get_models():
        return {
            'random': RandomCompetitorModel,
            'follower': FollowerCompetitorModel,
            'leader': LeaderCompetitorModel,
            'cyclic': CyclicCompetitorModel,
            'aggressive': AggressiveCompetitorModel,
        }


class RandomCompetitorModel(CompetitorModel):
    """Random uniform pricing within a range"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.min_price = self.params.get('min_price', 1)
        self.max_price = self.params.get('max_price', 100)
        
    def get_price(self, period, history=None, our_price=None):
        return np.random.randint(self.min_price, self.max_price + 1)


class FollowerCompetitorModel(CompetitorModel):
    """Follower that adjusts prices based on our previous price"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.lag = self.params.get('lag', 1)
        self.adjustment_factor = self.params.get('adjustment_factor', 0.95)
        self.min_price = self.params.get('min_price', 1)
        self.max_price = self.params.get('max_price', 100)
        self.initial_price = self.params.get('initial_price', 50)
        
    def get_price(self, period, history=None, our_price=None):
        if history is None or len(history) < self.lag:
            return self.initial_price
            
        # Follow our price from lag periods ago
        our_previous_price = history[-self.lag][0]
        price = int(our_previous_price * self.adjustment_factor)
        return np.clip(price, self.min_price, self.max_price)


class LeaderCompetitorModel(CompetitorModel):
    """Price leader that sets a trend"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.trend_type = self.params.get('trend_type', 'increasing')  # 'increasing', 'decreasing', 'cyclic'
        self.trend_factor = self.params.get('trend_factor', 0.02)
        self.min_price = self.params.get('min_price', 1)
        self.max_price = self.params.get('max_price', 100)
        self.initial_price = self.params.get('initial_price', 50)
        
    def get_price(self, period, history=None, our_price=None):
        if history is None or len(history) == 0:
            return self.initial_price
            
        last_price = history[-1][history[-1].index(max(history[-1])) if self.trend_type == 'highest' else 1]
        
        if self.trend_type == 'increasing':
            price = last_price * (1 + self.trend_factor)
        elif self.trend_type == 'decreasing':
            price = last_price * (1 - self.trend_factor)
        else:  # cyclic
            phase = (period % 20) / 20  # 20-period cycle
            price = last_price * (1 + self.trend_factor * np.sin(2 * np.pi * phase))
            
        return int(np.clip(price, self.min_price, self.max_price))


class CyclicCompetitorModel(CompetitorModel):
    """Competitor that cycles prices according to a pattern"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.cycle_length = self.params.get('cycle_length', 10)
        self.min_price = self.params.get('min_price', 20)
        self.max_price = self.params.get('max_price', 80)
        
    def get_price(self, period, history=None, our_price=None):
        # Sinusoidal price pattern
        cycle_position = period % self.cycle_length
        normalized_position = cycle_position / self.cycle_length
        amplitude = (self.max_price - self.min_price) / 2
        midpoint = (self.max_price + self.min_price) / 2
        price = midpoint + amplitude * np.sin(2 * np.pi * normalized_position)
        return int(price)


class AggressiveCompetitorModel(CompetitorModel):
    """Aggressive competitor that tries to undercut our price"""
    def __init__(self, params=None):
        super().__init__(params or {})
        self.undercutting_factor = self.params.get('undercutting_factor', 0.9)
        self.min_price = self.params.get('min_price', 1)
        self.max_price = self.params.get('max_price', 100)
        self.initial_price = self.params.get('initial_price', 50)
        
    def get_price(self, period, history=None, our_price=None):
        if our_price is None:
            return self.initial_price
            
        price = int(our_price * self.undercutting_factor)
        return np.clip(price, self.min_price, self.max_price)


def simulate(periods=100, num_competitors=3, create_plots=True, 
             demand_model_name='standard', demand_params=None,
             competitor_models=None):
    """
    Run a pricing simulation with the specified parameters
    
    Parameters:
    - periods: Number of periods to simulate
    - num_competitors: Number of competitors in the market
    - create_plots: Whether to create visualization plots
    - demand_model_name: Name of the demand model to use
    - demand_params: Parameters for the demand model
    - competitor_models: List of tuples (model_name, params) for each competitor
    """
    root = os.getcwd()
    prices_file = os.path.join(root, 'historical_prices.csv')
    demands_file = os.path.join(root, 'main_project', 'historical_demands.csv')
    
    # Create simulation output folder with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(root, 'simulation_output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup demand model
    demand_model_class = DemandModel.get_models().get(demand_model_name, StandardDemandModel)
    demand_model = demand_model_class(demand_params)
    
    # Setup competitor models
    if competitor_models is None:
        competitor_models = [('random', {}) for _ in range(num_competitors)]
    
    competitor_instances = []
    competitor_model_names = []
    for model_name, params in competitor_models:
        model_class = CompetitorModel.get_models().get(model_name, RandomCompetitorModel)
        competitor_instances.append(model_class(params))
        competitor_model_names.append(model_name)
    
    # Save simulation configuration
    config = {
        'timestamp': timestamp,
        'periods': periods,
        'num_competitors': num_competitors,
        'demand_model': {
            'name': demand_model_name,
            'params': demand_model.params
        },
        'competitor_models': [
            {'name': model_name, 'params': params}
            for model_name, params in competitor_models
        ]
    }
    
    config_df = pd.DataFrame({
        'parameter': list(config.keys()),
        'value': [str(v) for v in config.values()]
    })
    config_path = os.path.join(output_dir, 'simulation_config.csv')
    config_df.to_csv(config_path, index=False)
    
    # Initialize history variables
    price_history = []       # List of lists: [our_price, comp1, comp2, ...]
    our_demand_history = []  # List of our demand values
    revenue_history = []     # List of our revenue values

    for t in range(1, periods + 1):
        # Write historical_prices.csv (previous periods only)
        if price_history:
            cols = ['team1'] + [f'team{i}' for i in range(2, len(competitor_instances) + 2)]
            df_prices = pd.DataFrame(price_history, columns=cols)
            df_prices.insert(0, 'period', range(1, len(df_prices) + 1))
            df_prices.to_csv(prices_file, index=False)

        # Write historical_demands.csv for our team (previous periods only)
        if our_demand_history:
            df_dem = pd.DataFrame({
                'period': list(range(1, len(our_demand_history) + 1)),
                'price': [row[0] for row in price_history],
                'demand': our_demand_history
            })
            df_dem.to_csv(demands_file, index=False)

        # Invoke strategy to pick price for this period
        os.chdir(os.path.join(root, 'main_project'))
        our_price = strategy()
        os.chdir(root)

        # Get competitor prices using their models
        comp_prices = []
        for competitor in competitor_instances:
            comp_price = competitor.get_price(t, price_history, our_price)
            comp_prices.append(comp_price)
            
        all_prices = [our_price] + comp_prices
        price_history.append(all_prices)

        # Get demand for our price using the demand model
        our_demand = demand_model.get_demand(our_price, t, our_demand_history)
        our_demand_history.append(our_demand)

        revenue = our_price * our_demand
        revenue_history.append(revenue)
        print(f"Period {t:3d}: Price = {our_price:3d}, Demand = {our_demand:3d}, Revenue = {revenue:6d}")

    # Final write to include last period
    if price_history:
        cols = ['team1'] + [f'team{i}' for i in range(2, len(competitor_instances) + 2)]
        df_prices = pd.DataFrame(price_history, columns=cols)
        df_prices.insert(0, 'period', range(1, len(df_prices) + 1))
        df_prices.to_csv(prices_file, index=False)
    if our_demand_history:
        df_dem = pd.DataFrame({
            'period': list(range(1, len(our_demand_history) + 1)),
            'price': [row[0] for row in price_history],
            'demand': our_demand_history
        })
        df_dem.to_csv(demands_file, index=False)
    
    # Save simulation results to CSV files in output directory
    periods_array = np.arange(1, periods + 1)
    our_prices = np.array([row[0] for row in price_history])
    
    # Create detailed results dataframe and save to CSV
    results_df = pd.DataFrame({
        'period': periods_array,
        'our_price': our_prices,
        'demand': our_demand_history,
        'revenue': revenue_history,
    })
    
    # Add competitor prices to the dataframe
    for i in range(len(competitor_instances)):
        results_df[f'competitor_{i+1}_price'] = [row[i+1] for row in price_history]
        results_df[f'competitor_{i+1}_model'] = competitor_model_names[i]
    
    # Save results to CSV
    results_csv_path = os.path.join(output_dir, 'simulation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    # Save summary statistics to CSV
    summary_df = pd.DataFrame({
        'metric': ['average_price', 'average_demand', 'average_revenue', 'total_revenue', 'periods'],
        'value': [
            np.mean(our_prices),
            np.mean(our_demand_history),
            np.mean(revenue_history),
            np.sum(revenue_history),
            periods
        ]
    })
    
    summary_csv_path = os.path.join(output_dir, 'simulation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    total_revenue = sum(revenue_history)
    print(f"\nSimulation complete over {periods} periods.")
    print(f"Total revenue: {total_revenue}")
    print(f"Results saved to: {output_dir}")
    
    # Create plots if requested
    if create_plots:
        plot_simulation_results(price_history, our_demand_history, revenue_history, 
                                len(competitor_instances), output_dir, demand_model_name, 
                                competitor_model_names)


def plot_simulation_results(price_history, our_demand_history, revenue_history, 
                           num_competitors, output_dir, demand_model_name="standard",
                           competitor_model_names=None):
    """
    Create plots of simulation results:
    1. Prices over time (our price vs competitors)
    2. Our demand over time
    3. Our revenue over time
    4. Price-demand relationship
    """
    periods = list(range(1, len(price_history) + 1))
    price_array = np.array(price_history)
    
    # Use default competitor names if not provided
    if competitor_model_names is None:
        competitor_model_names = [f"Competitor {i+1}" for i in range(num_competitors)]
    else:
        # Make sure we have the right number of names
        while len(competitor_model_names) < num_competitors:
            competitor_model_names.append(f"Competitor {len(competitor_model_names)+1}")
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Prices over time
    plt.subplot(2, 2, 1)
    plt.plot(periods, price_array[:, 0], 'b-', linewidth=2, label='Our Strategy')
    for i in range(1, min(num_competitors + 1, price_array.shape[1])):
        plt.plot(periods, price_array[:, i], '--', alpha=0.6, 
                label=f'{competitor_model_names[i-1].capitalize()}')
    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.title(f'Price Strategy Over Time (Demand Model: {demand_model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Our demand over time
    plt.subplot(2, 2, 2)
    plt.plot(periods, our_demand_history, 'g-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Demand')
    plt.title(f'Our Demand Over Time (Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Our revenue over time
    plt.subplot(2, 2, 3)
    plt.plot(periods, revenue_history, 'r-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Revenue')
    plt.title(f'Our Revenue Over Time (Demand Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Price-Demand relationship (scatter)
    plt.subplot(2, 2, 4)
    plt.scatter(price_array[:, 0], our_demand_history, c=periods, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Period')
    plt.xlabel('Our Price')
    plt.ylabel('Demand')
    plt.title(f'Price-Demand Relationship (Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots individually
    plt.figure(figsize=(10, 6))
    plt.plot(periods, price_array[:, 0], 'b-', linewidth=2, label='Our Strategy')
    for i in range(1, min(num_competitors + 1, price_array.shape[1])):
        plt.plot(periods, price_array[:, i], '--', alpha=0.6, 
                label=f'{competitor_model_names[i-1].capitalize()}')
    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.title(f'Price Strategy Over Time (Demand Model: {demand_model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prices_over_time.png'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(periods, our_demand_history, 'g-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Demand')
    plt.title(f'Our Demand Over Time (Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_over_time.png'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(periods, revenue_history, 'r-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Revenue')
    plt.title(f'Our Revenue Over Time (Demand Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_over_time.png'))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(price_array[:, 0], our_demand_history, c=periods, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Period')
    plt.xlabel('Our Price')
    plt.ylabel('Demand')
    plt.title(f'Price-Demand Relationship (Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_demand_relationship.png'))
    
    # Create competitor price distribution plot
    plt.figure(figsize=(12, 6))
    box_data = [price_array[:, i] for i in range(price_array.shape[1])]
    box_labels = ['Our Strategy'] + [model.capitalize() for model in competitor_model_names]
    plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    plt.ylabel('Price')
    plt.title(f'Price Distribution Comparison (Demand Model: {demand_model_name})')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'competitor_price_distribution.png'))
    
    # Save combined plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(periods, price_array[:, 0], 'b-', linewidth=2, label='Our Strategy')
    for i in range(1, min(num_competitors + 1, price_array.shape[1])):
        plt.plot(periods, price_array[:, i], '--', alpha=0.6, 
                label=f'{competitor_model_names[i-1].capitalize()}')
    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.title(f'Price Strategy Over Time (Demand Model: {demand_model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(periods, our_demand_history, 'g-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Demand')
    plt.title(f'Our Demand Over Time (Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(periods, revenue_history, 'r-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Revenue')
    plt.title(f'Our Revenue Over Time (Demand Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(price_array[:, 0], our_demand_history, c=periods, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Period')
    plt.xlabel('Our Price')
    plt.ylabel('Demand')
    plt.title(f'Price-Demand Relationship (Model: {demand_model_name})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_plots.png'))
    
    # Create a moving average revenue plot
    plt.figure(figsize=(10, 6))
    window_size = min(10, len(revenue_history))
    revenue_ma = np.convolve(revenue_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(periods[window_size-1:], revenue_ma, 'r-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Revenue (10-period Moving Average)')
    plt.title(f'Revenue Trend (Moving Average) - Demand Model: {demand_model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_moving_average.png'))


def parse_arguments():
    """Parse command line arguments for simulator"""
    parser = argparse.ArgumentParser(description='Run pricing strategy simulation')
    parser.add_argument('--periods', type=int, default=100, help='Number of periods to simulate')
    parser.add_argument('--competitors', type=int, default=3, help='Number of competitors')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--demand-model', choices=list(DemandModel.get_models().keys()), default='standard',
                        help='Demand model to use')
    parser.add_argument('--seasonal-amplitude', type=float, default=0.2, 
                        help='Amplitude for seasonal demand (if seasonal model used)')
    parser.add_argument('--seasonal-period', type=int, default=20, 
                        help='Period length for seasonal demand (if seasonal model used)')
    parser.add_argument('--competitor-models', type=str, default='random,random,random',
                        help='Comma-separated list of competitor models')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Configure demand model parameters
    demand_params = {}
    if args.demand_model == 'seasonal':
        demand_params = {
            'amplitude': args.seasonal_amplitude,
            'period_length': args.seasonal_period
        }
    
    # Configure competitor models
    competitor_model_names = args.competitor_models.split(',')
    competitor_models = []
    
    for i, model_name in enumerate(competitor_model_names):
        if model_name == 'follower':
            params = {'lag': 1, 'adjustment_factor': 0.95}
        elif model_name == 'leader':
            params = {'trend_type': 'increasing' if i % 2 == 0 else 'decreasing'}
        elif model_name == 'cyclic':
            params = {'cycle_length': 10 + i * 5}
        elif model_name == 'aggressive':
            params = {'undercutting_factor': 0.9 - i * 0.05}
        else:  # random
            params = {}
        
        competitor_models.append((model_name, params))
    
    # Ensure we have enough competitor models
    while len(competitor_models) < args.competitors:
        competitor_models.append(('random', {}))
    
    # Trim to requested number
    competitor_models = competitor_models[:args.competitors]
    
    simulate(
        periods=args.periods,
        num_competitors=args.competitors,
        create_plots=not args.no_plots,
        demand_model_name=args.demand_model,
        demand_params=demand_params,
        competitor_models=competitor_models
    )

# python simulator_main_project.py --periods 50 --demand-model seasonal --competitor-models follower,aggressive,cyclic