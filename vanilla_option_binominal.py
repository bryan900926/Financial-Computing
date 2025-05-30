import sys
import os
import importlib.util
import numpy as np
from scipy.stats import norm  
path_to_file = "C:/Users/bryan/OneDrive/桌面/金融計算/hw1/hw1.py"
spec = importlib.util.spec_from_file_location("hw1", path_to_file)
hw1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw1)

def bs_formula(option_type: str, S: int, r: int, T: float, sigma: float, K: int, q: int, output = False): 
    d1 = (np.log(S / K) + (r - q + 0.5 * np.square(sigma)) * T) / (sigma * np.sqrt(T)) 
    d2 = d1 -  sigma * np.sqrt(T)
    price = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            if option_type == "call"
            else K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))
    print("------bs model-----------")
    print(f"price of option: {price}")
    return price

def payoff_put(S_vector: int, K: int)-> int:
    return np.maximum(K - S_vector, 0)

def payoff_call(S_vector: int, K: int) -> int:
    return np.maximum(S_vector - K, 0)

def binomial_option_price(option_type: str, S: float, K: float, T: float, r: float,
                          sigma: float, N: int, q: float, payoff_fuct, output = False) -> float:
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    #space O(N^2)
    stock_tree = np.zeros((N + 1, N + 1))
    option_tree = np.zeros((N + 1, N + 1))

    stock_tree[0, 0] = S

    # for i in range(N):
    #     for j in range(i + 1):
    #         stock_tree[j, i + 1] = stock_tree[j, i] * u
    #         stock_tree[j + 1, i + 1] = stock_tree[j, i] * d

    for i in range(N):
        stock_tree[:i + 1, i + 1] = stock_tree[:i + 1, i] * u
        stock_tree[i + 1, i + 1] = stock_tree[i, i] * d

    
    # for j in range(N + 1):
    #     option_tree[j][N] = payoff_fuct(stock_tree[j][N], K)

    # for i in range(N - 1, -1, -1):
    #     for j in range(i + 1):
    #         expect_payoff = np.exp(-r * dt) * (p * option_tree[j][i + 1] + (1 - p) * option_tree[j + 1][i + 1])
    #         excerse_now = payoff_fuct(stock_tree[j][i], K)
    #         option_tree[j][i] = (max(expect_payoff, excerse_now) if option_type == 'A'
    #                              else expect_payoff)
    # space optimization
    option_tree = np.array([payoff_fuct(S * pow(u, N - j) * pow(d, j), K) for j in range(N + 1)])
    # for i in range(N - 1, -1, -1):
    #     for j in range(i + 1):
    #         expect_payoff = np.exp(-r * dt) * (p * option_tree[j] + (1 - p) * option_tree[j + 1])
    #         excerse_now = payoff_fuct(stock_tree[j][i], K)
    #         option_tree[j] = (max(expect_payoff, excerse_now) if option_type == 'A'
    #                              else expect_payoff)
    for i in range(N - 1, -1, -1):
        option_tree = np.exp(-r * dt) * (p * option_tree[:-1] + (1 - p) * option_tree[1:])
        excerse_now = payoff_fuct(stock_tree[:i + 1, i], K)
        option_tree = (np.maximum(option_tree[:i + 1], excerse_now) if option_type == 'A'
                                 else option_tree)
    if output:    
        print("---------Binomial model------------")
        print(option_tree[0])
    return option_tree[0]
            
def combinatorial_CRR(S: float, K: float, T: float, r: float,
                      sigma: float, N: int, q: float, payoff_fuct, output = False) -> float:
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    presum_factorial = [0] * (N + 1)
    presum_u = [0] * (N + 1)
    presum_d = [0] * (N + 1)

    for i in range(1, N + 1):
        presum_factorial[i] += presum_factorial[i - 1] + np.log(i)
        presum_u[i] += presum_u[i - 1] + np.log(u)
        presum_d[i] += presum_d[i - 1] + np.log(d)

    def helper(n, j, p, S0):
        ln_prob = 0 
        ln_price = np.log(S0)
        ln_prob += presum_factorial[n] - presum_factorial[j] - presum_factorial[n - j]
        ln_prob += (n - j) * np.log(p) + j * np.log(1 - p)
        prob = np.exp(ln_prob) 
        ln_price += presum_u[n - j] + presum_d[j]
        price = np.exp(ln_price)
        
        return prob * payoff_fuct(price, K)
    
    value = 0
    for j in range(N + 1):
        value += helper(N, j, p, S)
    if output:   
       print("--------CRR model------------")
       print(np.exp(-r * T) * value )
    return np.exp(-r * T) * value  

def main():     
    # price = bs_formula(S = 50, r = 0.1, q = 0.05, T = 0.5, sigma = 0.4, K = 50, option_type = "call", output = True)
    # price = bs_formula(S = 50, r = 0.1, q = 0.05, T = 0.5, sigma = 0.4, K = 50, option_type = "put", output = True)
    # confidence_interval = hw1.monte_carlo(S0 = 50, r = 0.1, q = 0.05, T = 0.4, sigma = 0.5, 
    #             K1 = 50, sample_size = 10000, iterations = 20, k = 1.96, payoff_fuct = payoff_call, output = True)   
    # p = binomial_option_price(option_type = "", S = 50, r = 0.1, q = 0.05, T = 0.5, 
    #                       sigma = 0.4, K = 50, payoff_fuct = payoff_call, N = 500, output = True)
    # p = binomial_option_price(option_type = "A", S = 50, r = 0.1, q = 0.05, T = 0.5, 
    #                       sigma = 0.4, K = 50, payoff_fuct = payoff_call, N = 500, output = True)
    # p = binomial_option_price(option_type = "", S = 50, r = 0.1, q = 0.05, T = 0.5, 
    #                       sigma = 0.4, K = 50, payoff_fuct = payoff_put, N = 500, output = True)
    p = binomial_option_price(option_type = "A", S = 50, r = 0.1, q = 0.05, T = 0.5, 
                          sigma = 0.4, K = 50, payoff_fuct = payoff_put, N = 1000, output = True)
    # combinatorial_CRR(S = 50, r = 0.1, q = 0.05, T = 0.5, 
    #                       sigma = 0.4, K = 50, payoff_fuct = payoff_put, N = 100000, output = True)

if __name__ == '__main__':
    main()
