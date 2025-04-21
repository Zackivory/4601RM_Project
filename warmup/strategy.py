def strategy():
    import numpy as np, pandas as pd
    from scipy.optimize import minimize
    try:
        data=pd.read_csv("historical_data.csv", usecols=["p_t", "X_t"])
        p_hist=data["p_t"].to_numpy()
        x_hist=data["X_t"].to_numpy()
    except FileNotFoundError:
        return 10.0  # Start with a lower initial price
    if data.empty:
        return 10.0

    # If we have recent rejections, be more aggressive in lowering price
    if len(x_hist) > 0 and np.mean(x_hist[-5:]) == 0:  # Last 5 were all rejections
        return max(0.01, np.mean(p_hist[-5:]) * 0.7)  # Reduce price by 30%

    def nll(beta_vec):
        beta = float(beta_vec[0])
        prob = 1/(1+ np.exp(beta * p_hist))
        eps  = 1e-12
        return -(x_hist*np.log(prob+eps) +(1-x_hist)*np.log(1-prob+eps)).sum()

    beta_hat = minimize(nll, x0=np.array([0.5]), bounds=[(1e-3, 20.0)]).x[0]

    def neg_rev(p_vec):
        p=float(p_vec[0])
        prob =1/(1+np.exp(beta_hat*p))
        return -p*prob
    opt_p = minimize(neg_rev, x0=np.array([10.0]), bounds=[(0.01, 200.0)]).x[0]
    
    # Add exploration in early rounds
    if len(data) < 20:  # first 20 customers
        opt_p = max(0.01, opt_p * (1 + np.random.normal(0, 0.2)))
    
    return float(opt_p)
if __name__ == '__main__':
    print(strategy())