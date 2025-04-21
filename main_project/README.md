# Dynamic Pricing Strategy System

This project implements an advanced dynamic pricing strategy system that adapts to market conditions using machine learning techniques and competitor analysis.

## Overview

The system uses a combination of different pricing strategies that are weighted based on their historical performance. The weights are updated using a multiplicative weight update algorithm, which gradually shifts more weight to better-performing strategies.

## Strategies

The system includes the following pricing strategies:

1. **Random Exploration** - Sets a random price to explore the demand curve.
2. **Mean Price Follower** - Matches the average market price from all competitors.
3. **Regression Optimal** - Fits a linear demand curve to historical price-demand data and calculates the revenue-maximizing price.
4. **Upper Confidence Bound (UCB)** - Uses a multi-armed bandit approach to balance exploration and exploitation of pricing options.
5. **Competitive Response** - Analyzes competitor pricing trends and adjusts price accordingly (undercutting in high-price markets, raising in low-price markets).

### Regression Optimal Pricing

This strategy fits a linear demand curve to historical data:

$$d \approx a + b \cdot p$$

Where $d$ is demand and $p$ is price. Revenue is calculated as:

$$R(p) = p \cdot d = p \cdot (a + b \cdot p) = a \cdot p + b \cdot p^2$$

To find the revenue-maximizing price, we take the derivative and set it to zero:

$$\frac{dR}{dp} = a + 2b \cdot p = 0$$

Solving for $p$ gives us the optimal price:

$$p^* = -\frac{a}{2b}$$

This is valid when $b < 0$ (decreasing demand with increasing price).

### Upper Confidence Bound (UCB)

The UCB strategy balances exploration and exploitation using the UCB1 algorithm. For each price $p$, we calculate:

$$\text{UCB}(p) = \bar{r}_p + c\sqrt{\frac{\ln(T)}{n_p}}$$

Where:
- $\bar{r}_p$ is the average revenue for price $p$
- $n_p$ is the number of times price $p$ has been used
- $T$ is the total number of periods
- $c$ is the exploration coefficient (set to 2.0)

## Weighted Average Mechanism

The system combines all strategies using a weighted average approach, where:
- Weights are initialized equally
- After each pricing period, weights are updated based on how well each strategy's recommendation would have performed
- The multiplicative weight update algorithm increases weights for better strategies
- Weights persist across sessions in a CSV file

### Multiplicative Weight Update

For each strategy $i$, the weight update follows:

$$w_i \leftarrow w_i \cdot (1 + \eta)^{r_i}$$

Where:
- $w_i$ is the weight for strategy $i$
- $\eta$ is the learning rate (set to 0.1)
- $r_i$ is the reward ratio (estimated revenue / actual revenue)

After updating, weights are normalized:

$$w_i \leftarrow \frac{w_i}{\sum_j w_j}$$

## Implementation Details

### Data Sources
- Historical prices for all teams are read from `historical_prices.csv` in the parent directory
- Historical demand data is read from `historical_demands.csv`
- Strategy weights are saved and loaded from `strategy_weights.csv`

### Adapting to Market Conditions
- In high-price markets (average >60), the competitive strategy slightly undercuts the market
- In low-price markets (average <30), it sets slightly higher prices to maintain profitability
- In medium-price markets, it matches the average market price

## Data Structures

The system uses the following CSV file formats:

### historical_prices.csv
```
period,team1,team2,team3,...,teamN
1,45.0,42.5,39.8,...,41.2
2,46.5,42.0,38.5,...,43.7
...
```
Each row represents a period, with the first column being the period number and subsequent columns containing the price set by each team.

### historical_demands.csv
```
period,price,demand
1,45.0,120
1,42.5,132
1,39.8,145
...
```
Each row contains the period, price set, and resulting demand observed. Multiple entries exist for each period (one per team).

### strategy_weights.csv
This csv is create and manged by our `strategy()` function automatically. The file is initially created with equal weights for all strategies if it doesn't exist. After each pricing period, the `strategy()` function updates the weights based on performance and saves the changes back to this file. This ensures that the learning persists across different runs of the application. If the file becomes corrupted or needs to be reset, simply delete it and the system will reinitialize the weights evenly on the next run.
```
strategy,weight
Random,0.12
MeanPrice,0.22
Regression,0.35
UCB,0.18
Competitive,0.13
```
Contains the name of each strategy and its corresponding weight used in the weighted average calculation.

## Usage

The strategy function is called to determine the optimal price for each period. It handles all data loading, strategy execution, and weight updates automatically.

```python
from strategy import strategy

# Get the recommended price for this period
recommended_price = strategy()
```

## Extensibility

The system is designed to be extensible. To add a new strategy:
1. Implement the strategy function with the same parameter signature as existing strategies
2. Add the strategy to the list in the `weighted_average` function
3. Add a name for the strategy in the `strat_names` list 