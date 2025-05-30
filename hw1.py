import numpy as np
from scipy.stats import norm   

def pricing_hw1(S0: int, r: int, q: int, T: int, sigma: float, K1: int, K2: int, K3: int, K4: int) -> str: 

    d1 = (np.log(K1 / S0) - (r - q + 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(K2 / S0) - (r - q + 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d11 = (np.log(K1 / S0) - (r - q - 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d22 = (np.log(K2 / S0) - (r - q - 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d3 = (np.log(K3 / S0) - (r - q + 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d4 = (np.log(K4 / S0) - (r - q + 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d33 = (np.log(K3 / S0) - (r - q - 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))
    d44 = (np.log(K4 / S0) - (r - q - 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T))

    a = S0 * np.exp((r - q) * T) * (norm.cdf(d2, loc = 0, scale = 1) - norm.cdf(d1, loc = 0, scale = 1))
    b = K1 * (norm.cdf(d22, loc = 0, scale = 1) - norm.cdf(d11, loc = 0, scale = 1))
    c = (K2 - K1) * (norm.cdf(d33, loc = 0, scale = 1) - norm.cdf(d22, loc = 0, scale = 1))
    d = K4 * (norm.cdf(d44, loc = 0, scale = 1) - norm.cdf(d33, loc = 0, scale = 1))
    e = S0 * np.exp((r - q) * T) * (norm.cdf(d4, loc = 0, scale = 1) - norm.cdf(d3, loc = 0, scale = 1))
    payoff = a - b + c + (K2 - K1) / (K4 - K3) * (d - e) 
    price = np.exp(-r * T) * payoff 
    return f"Price: {price}"
# print(pricing_hw1(S0 = 50, r = 0.01, q = 0.005, T = 1, sigma = 0.25, K1 = 60, K2 = 70, K3 = 80, K4 = 90))
def monte_carlo(S0: int, r: int, q: int,  T: int, sigma: float, 
                K1: int, sample_size: int, iterations: int, k: int, payoff_fuct, output = False) -> str:

    mean, variance = np.log(S0) + (r - q - 0.5 * sigma ** 2) * T, np.square(sigma) * T
    price_vector = np.zeros(iterations)

    for j in range(iterations):
        cum_payoff = 0
        for _ in range(sample_size):  
            ln_St = mean + np.random.randn() * np.sqrt(variance)
            St = np.exp(ln_St)
            cum_payoff +=  payoff_fuct(St, K1) / sample_size
        price_vector[j] = cum_payoff * np.exp(-r * T)
    mu, std = np.mean(price_vector), np.std(price_vector)
    if output:   
      print("--------- monde carlo-----------")
      print(f"mean: {mu}  std: {std}")
      print(f"Confidence Interval: [{mu - 2 * std}, {mu + 2 * std}]")

def hw1_payoff(St, K1, K2, K3, K4):
    if St < K1 or St > K4:
        return 0
    elif K1 <= St < K2: return St - K1
    elif K2 <= St <= K3: return K2 - K1
    return (K4 - St) * (K2 - K1) / (K4 - K3)

def main():     
    price = pricing_hw1(S0 = 100, r = 0.05, q = 0.02, T = 0.4, sigma = 0.5, 
                K1 = 90, K2 = 98, K3 = 102, K4 = 104)
    print(price)
    confidence_interval = monte_carlo(S0 = 100, r = 0.05, q = 0.02, T = 0.4, sigma = 0.5, 
                K1 = 90, K2 = 98, K3 = 102, K4 = 104, sample_size = 10000, iterations = 20, k = 1.96)
    print(confidence_interval)
    
if __name__ == '__main__':
    main()
