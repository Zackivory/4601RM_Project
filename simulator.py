import numpy as np
import os
from warmup.strategy import strategy

def simulate_rounds(num_rounds=1000, true_beta=0.5, seed=42):
    """
    Simulate multiple rounds of pricing and customer responses.
    
    Args:
        num_rounds: Number of rounds to simulate
        true_beta: The true value of beta parameter
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Initialize historical data file
    if os.path.exists('warmup/historical_data.csv'):
        os.remove('warmup/historical_data.csv')

    # Write header row
    with open('warmup/historical_data.csv', 'w') as f:
        f.write("p_t,X_t\n")

    total_revenue = 0
    acceptance_rate = 0

    print(f"Simulating {num_rounds} rounds with true beta = {true_beta}")
    print("-" * 50)

    for t in range(num_rounds):
        # Get price from strategy
        p_t = strategy()

        # Calculate acceptance probability
        prob_accept = np.exp(-true_beta * p_t) / (1 + np.exp(-true_beta * p_t))

        # Simulate customer response
        X_t = np.random.binomial(1, prob_accept)

        # Record the outcome
        with open('warmup/historical_data.csv', 'a') as f:
            f.write(f"{p_t},{X_t}\n")

        # Update statistics
        if X_t:
            total_revenue += p_t
            acceptance_rate += 1

    # Calculate final statistics
    acceptance_rate /= num_rounds
    avg_revenue = total_revenue / num_rounds

    print(f"Simulation Results:")
    print(f"Total Revenue: {total_revenue:.2f}")
    print(f"Average Revenue per Round: {avg_revenue:.2f}")
    print(f"Acceptance Rate: {acceptance_rate:.2%}")

    return total_revenue, avg_revenue, acceptance_rate

if __name__ == "__main__":
    # Run simulation with different beta values to test strategy's adaptability
    beta_values = [0.01,0.1, 0.5]
    results = []

    for beta in beta_values:
        print(f"\nTesting with beta = {beta}")
        total_rev, avg_rev, acc_rate = simulate_rounds(true_beta=beta)
        results.append({
            'beta': beta,
            'total_revenue': total_rev,
            'avg_revenue': avg_rev,
            'acceptance_rate': acc_rate
        })

    # Print summary table
    print("\nSummary of Results:")
    print("-" * 50)
    print("Beta\tTotal Revenue\tAvg Revenue\tAcceptance Rate")
    print("-" * 50)
    for r in results:
        print(f"{r['beta']}\t{r['total_revenue']:.2f}\t\t{r['avg_revenue']:.2f}\t\t{r['acceptance_rate']:.2%}")