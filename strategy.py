

def strategy():
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    # Read historical data
    try:
        data = pd.read_csv('historical_data.csv', header=None, names=['price', 'accepted'])
    except FileNotFoundError:
        # If no historical data exists yet, start with an initial price
        return 1.0  # Initial price
    
    if len(data) == 0:
        return 1.0  # Initial price if no data yet
    
    # Maximum likelihood estimation of beta
    def neg_log_likelihood(beta):
        prices = data['price'].values
        accepted = data['accepted'].values
        probs = np.exp(-beta * prices) / (1 + np.exp(-beta * prices))
        return -np.sum(accepted * np.log(probs) + (1 - accepted) * np.log(1 - probs))
    
    # Find beta that minimizes negative log likelihood
    result = minimize(neg_log_likelihood, x0=1.0, bounds=[(0.001, 10.0)])
    beta_hat = result.x[0]
    
    # Compute optimal price based on estimated beta
    # The optimal price maximizes p * (e^(-beta*p) / (1 + e^(-beta*p)))
    def revenue(p):
        return -p * (np.exp(-beta_hat * p) / (1 + np.exp(-beta_hat * p)))
    
    # Find price that maximizes revenue
    result = minimize(revenue, x0=1.0, bounds=[(0.01, 100.0)])
    optimal_price = result.x[0]
    
    # Add some exploration if we have limited data
    if len(data) < 10:
        # Add small random perturbation to explore
        optimal_price *= (1 + np.random.normal(0, 0.1))
    
    return optimal_price
