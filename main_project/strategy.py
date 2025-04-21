import pandas as pd
import numpy as np
import os


def strategy():
    # ─── load history ─────────────────────────────────────────────
    prices_path = os.path.join(os.pardir, 'historical_prices.csv')
    if os.path.exists(prices_path):
        # Properly read all team prices - each row is a period, each column is a team
        prices_df = pd.read_csv(prices_path)
        # Skip the period column and use only price columns
        if 'period' in prices_df.columns:
            prices_df = prices_df.drop(columns=['period'])
        all_team_prices = prices_df.values
        # Our prices are in the first column (index 0)
        our_prices = all_team_prices[:, 0:1] if all_team_prices.size > 0 else np.empty((0,))
        price_hist = all_team_prices  # Keep all teams' prices for market analysis
    else:
        price_hist = np.empty((0,))
        our_prices = np.empty((0,))

    demand_path = 'historical_demands.csv'
    if os.path.exists(demand_path):
        demands_df = pd.read_csv(demand_path)
        # Get only the demand column
        if 'demand' in demands_df.columns:
            demand_hist = demands_df['demand'].values
        else:
            demand_hist = np.empty((0,))
    else:
        demand_hist = np.empty((0,))
        
    # Path for saving and loading weights
    weights_path = 'strategy_weights.csv'

    # ─── inner strategies ────────────────────────────────────────
    def random_explore(prices=None, demands=None):
        """Try a uniformly random price to explore demand."""
        return np.random.randint(1, 101)

    def follow_mean_price(prices=None, demands=None):
        """Match the average market price from last period."""
        if prices is None:
            prices = price_hist
            
        if prices.size == 0:
            return 50
        
        # Use the most recent period's prices from all teams
        last_period_prices = prices[-1]
        
        # Check for any non-numeric values and handle them
        numeric_prices = []
        for p in last_period_prices:
            try:
                numeric_prices.append(float(p))
            except (ValueError, TypeError):
                continue  # Skip non-numeric values
                
        if not numeric_prices:
            return 50  # Default if no valid prices
            
        # Calculate the mean price across all teams
        avg = np.mean(numeric_prices)
        return int(np.clip(round(avg), 1, 100))

    def regression_optimal(prices=None, demands=None, our_prices_param=None):
        """
        Fit a linear demand curve on your own data:
           d ≈ a + b·p
        Then maximize revenue R(p)=p·d => p* (a + b p).
        Optimum at p* = –a/(2b) (if b<0).
        """
        if prices is None:
            prices = price_hist
        if demands is None:
            demands = demand_hist
        if our_prices_param is None:
            our_prices_param = our_prices
            
        if len(demands) < 5 or our_prices_param.size == 0:
            return follow_mean_price(prices)

        # Use only our prices for regression
        your_prices = our_prices_param.flatten()
        # Safely convert to numeric values
        numeric_prices = []
        numeric_demands = []
        
        # Make sure we have matching valid price and demand pairs
        for i, p in enumerate(your_prices):
            if i < len(demands):
                try:
                    price_val = float(p)
                    demand_val = float(demands[i])
                    numeric_prices.append(price_val)
                    numeric_demands.append(demand_val)
                except (ValueError, TypeError):
                    continue
                    
        # Convert to numpy arrays
        your_prices = np.array(numeric_prices)
        y = np.array(numeric_demands)
        
        if len(y) < 5:  # Not enough valid data points
            return follow_mean_price(prices)
            
        X = np.vstack([np.ones_like(y), your_prices]).T
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coeffs
        if b >= 0:
            return follow_mean_price(prices)
        p_opt = -a / (2 * b)
        return int(np.clip(round(p_opt), 1, 100))

    def upper_confidence_bound(prices=None, demands=None, our_prices_param=None, c=2.0):
        """
        Implements a UCB1 strategy over discrete price arms {1,...,100},
        using past own price-demand history to estimate revenue per price.
        """
        if prices is None:
            prices = price_hist
        if demands is None:
            demands = demand_hist
        if our_prices_param is None:
            our_prices_param = our_prices
            
        T = len(demands)
        if T < 1:
            return random_explore()

        # Use only our prices for UCB
        flat_prices = our_prices_param.flatten()
        
        # Safely convert to numeric values
        numeric_prices = []
        numeric_demands = []
        
        # Make sure we have matching valid price and demand pairs
        for i, p in enumerate(flat_prices):
            if i < len(demands):
                try:
                    price_val = float(p)
                    demand_val = float(demands[i])
                    numeric_prices.append(price_val)
                    numeric_demands.append(demand_val)
                except (ValueError, TypeError):
                    continue
        
        # If no valid data, return random price
        if not numeric_prices:
            return random_explore()
            
        # Calculate revenues
        our_prices_float = np.array(numeric_prices)
        demands_float = np.array(numeric_demands)
        revenues = our_prices_float * demands_float
        
        counts = {}
        sums = {}
        for i in range(len(numeric_prices)):
            p = int(our_prices_float[i])
            counts[p] = counts.get(p, 0) + 1
            sums[p] = sums.get(p, 0) + revenues[i]

        ucb_values = {}
        for price in range(1, 101):
            n = counts.get(price, 0)
            if n > 0:
                mean_rev = sums[price] / n
                bonus = np.sqrt(c * np.log(T) / n)
                ucb_values[price] = mean_rev + bonus
            else:
                # ensure exploration of untried prices
                ucb_values[price] = float('inf')

        # choose price with highest UCB
        p_ucb = max(ucb_values, key=ucb_values.get)
        return p_ucb
        
    def competitive_response(prices=None, demands=None):
        """
        Analyze competitor pricing trends to determine optimal response.
        """
        if prices is None or prices.size == 0:
            return 50
            
        if prices.shape[1] <= 1:  # No competitor data
            return follow_mean_price(prices)
            
        # Get most recent period's competitor prices
        last_period = prices[-1]
        competitor_prices = last_period[1:] if len(last_period) > 1 else []
        
        if len(competitor_prices) == 0:
            return follow_mean_price(prices)
            
        # Calculate competitor price statistics
        min_price = np.min(competitor_prices)
        max_price = np.max(competitor_prices)
        avg_price = np.mean(competitor_prices)
        
        # Strategy: slightly undercut the average competitor price
        # But not too much to avoid a price war
        if avg_price > 60:
            # High price market - slightly undercut
            return int(np.clip(round(avg_price * 0.95), 1, 100))
        elif avg_price < 30:
            # Low price market - set slightly higher to maintain profitability
            return int(np.clip(round(avg_price * 1.05), 1, 100))
        else:
            # Medium price market - match average
            return int(np.clip(round(avg_price), 1, 100))
        
    def weighted_average():
        """
        Compute a weighted average of all strategy outputs
        using weights based on historical performance.
        Uses multiplicative weight update algorithm to adjust weights.
        """
        strats = [random_explore, follow_mean_price, regression_optimal, upper_confidence_bound, competitive_response]
        strat_names = ['random', 'mean_price', 'regression', 'ucb', 'competitive']
        # Initialize weights or load from file
        if os.path.exists(weights_path):
            try:
                saved_weights = pd.read_csv(weights_path, index_col=0)
                weights = saved_weights.values.flatten()
                if len(weights) != len(strats):
                    weights = np.ones(len(strats)) / len(strats)
            except:
                weights = np.ones(len(strats)) / len(strats)
        else:
            weights = np.ones(len(strats)) / len(strats)  # Equal weights initially
        
        if len(demand_hist) >= 2 and price_hist.size >= 2:
            # Get last period's price and demand
            last_price = our_prices[-1][0] if our_prices.ndim > 1 else our_prices[-1]  # Our price from last period
            last_demand = demand_hist[-1]  # Resulting demand
            last_revenue = int(last_price) * last_demand
            
            # Calculate what each strategy would have recommended last period
            temp_price_hist = price_hist[:-1]
            temp_demand_hist = demand_hist[:-1]
            temp_our_prices = our_prices[:-1]
            
            last_recommendations = []
            for strat in strats:
                if strat.__name__ in ['regression_optimal', 'upper_confidence_bound']:
                    # These strategies need our prices specifically
                    rec = strat(prices=temp_price_hist, demands=temp_demand_hist, our_prices_param=temp_our_prices)
                else:
                    rec = strat(prices=temp_price_hist, demands=temp_demand_hist)
                last_recommendations.append(rec)
            
            # Calculate how good each recommendation would have been
            eta = 0.1  # Learning rate
            rewards = []
            
            for rec_price in last_recommendations:
                # Simple model: assume demand scales linearly with price difference
                # More sophisticated models could be used here
                price_ratio = float(rec_price) / (float(last_price) if float(last_price) > 0 else 1)
                # Inverse relationship between price and demand
                estimated_demand = last_demand * (2 - price_ratio) if price_ratio <= 2 else 0
                estimated_revenue = rec_price * estimated_demand
                
                # Calculate reward as ratio to actual revenue (bounded)
                reward_ratio = estimated_revenue / (last_revenue if last_revenue > 0 else 1)
                reward_ratio = min(max(reward_ratio, 0.5), 2.0)  # Bound between 0.5 and 2.0
                rewards.append(reward_ratio)
            
            # Multiplicative weight update
            weights = weights * np.power(1 + eta, np.array(rewards))
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Save updated weights
            pd.DataFrame(weights, index=strat_names).to_csv(weights_path)
        
        # Get recommendations from each strategy using current histories
        prices = []
        for strat in strats:
            if strat.__name__ in ['regression_optimal', 'upper_confidence_bound']:
                # These strategies need our prices specifically
                rec = strat(prices=price_hist, demands=demand_hist, our_prices_param=our_prices)
            else:
                rec = strat(prices=price_hist, demands=demand_hist)
            prices.append(rec)
        
        # Compute weighted average price
        weighted_price = sum(p * w for p, w in zip(prices, weights))
        
        # Print weights for debugging (can be removed in production)
        weight_info = ', '.join([f"{name}: {w:.2f}" for name, w in zip(strat_names, weights)])
        print(f"Strategy weights: {weight_info}")
        
        return int(np.clip(round(weighted_price), 1, 100))

    # ─── mix strategies ─────────────────────────────────────────
    # Use the multiplicative weight update strategy
    return weighted_average()
