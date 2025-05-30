import numpy as np 
from sortedcontainers import SortedSet
from bisect import bisect_left
from time import time

class max_node:
    def __init__(self):
        self.child = SortedSet()
        self.payoff = {}

    def generate_paypff(self, S_max, St, payoff_fuct):
        self.payoff[S_max] = payoff_fuct(St, S_max)

def payoff_put(S_vector: int, K: int)-> int:
    return np.maximum(K - S_vector, 0)

def payoff_call(S_vector: int, K: int)-> int:
    return np.maximum(S_vector - K, 0)

def binomial_lookback_option_price(option_type: str, S: float, K: float, T: float, r: float,
                          sigma: float, N: int, t: int, q: float, payoff_fuct, S_mx_t, optimize, output = False) -> float:
    start = time()
    dt = (T - t) / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    S_mx_t = np.round(S_mx_t, 4)

    stock_tree = np.zeros((N + 1, N + 1))
    stock_tree[0, 0] = S
    S_max_tree = [[max_node() for _ in range(N + 1)] for _ in range(N + 1)]
    S_max_tree[0][0].child.add(S_mx_t)

    discount = np.exp(-r * dt)

    for i in range(N):
        stock_tree[:i + 1, i + 1] = stock_tree[:i + 1, i] * u
        stock_tree[i + 1, i + 1] = stock_tree[i, i] * d

    stock_tree = np.round(stock_tree, 4)
    if not optimize:
        for dt in range(1, N + 1):
            for j in range(dt + 1):
                St = round(stock_tree[j, dt], 4)
                if j != dt:    
                    pa1 = S_max_tree[j][dt - 1].child
                    target_idx = bisect_left(pa1, St)
                    if target_idx == len(pa1):
                        S_max_tree[j][dt].child.add(St)
                    S_max_tree[j][dt].child.update(pa1[target_idx:])
                if j > 0:
                    pa2 = S_max_tree[j - 1][dt - 1].child
                    target_idx = bisect_left(pa2, St)
                    if target_idx == len(pa2):
                       S_max_tree[j][dt].child.add(St)
                    S_max_tree[j][dt].child.update(pa2[target_idx:])
    else:   
        idx = bisect_left(stock_tree[0,:], S_mx_t)
        for dt in range(N + 1):
            for j in range(dt + 1):
                up = dt - j
                d = max(up - j, 0)
                start_idx = max(d, idx)
                end_idx = up 
                child = S_max_tree[j][dt].child
                child.update(stock_tree[0, start_idx:end_idx + 1])
                if stock_tree[0, d] <= S_mx_t:
                    child.add(S_mx_t)
        
    for i in range(N + 1):
        for s_mx in S_max_tree[i][N].child:
            S_max_tree[i][N].generate_paypff(S_max = s_mx, St = stock_tree[i, N], payoff_fuct = payoff_fuct)

    for dt in range(N - 1, -1, -1):
        for i in range(dt + 1):
            St = stock_tree[i][dt]
            St_rounded_next = stock_tree[i][dt + 1]
            up_dict = S_max_tree[i][dt + 1].payoff
            down_dict = S_max_tree[i + 1][dt + 1].payoff
    
            for s_mx in S_max_tree[i][dt].child:
                up_payoff = up_dict.get(s_mx, up_dict.get(St_rounded_next, 0))
                down_payoff = down_dict.get(s_mx, 0)
                node_payoff = discount * (p * up_payoff + (1 - p) * down_payoff)
                if option_type == "A":
                    immediate_exercise = payoff_fuct(St, s_mx)
                    S_max_tree[i][dt].payoff[s_mx] = max(node_payoff, immediate_exercise)
                else:
                    S_max_tree[i][dt].payoff[s_mx] = node_payoff

    end = time()
    if output:
       print("--------- binominal method -----------")
       print(f"price: {S_max_tree[0][0].payoff[S_mx_t]}", f"taken time: {end - start} seconds")
    return S_max_tree[0][0].payoff[S_mx_t]

def monte_carlo_lookback(N, St, r, q, T, t, sigma, St_max, sample_size, iterations, payoff_func, output = False) -> str:
    start = time()
    dt = (T - t) / N
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    total_paths = iterations * sample_size
    Z = np.random.randn(total_paths, N)

    ln_St_paths = np.log(St) + np.cumsum(drift + diffusion * Z, axis=1)
    ln_mx_paths = np.hstack((np.full((total_paths, 1), np.log(St_max)), ln_St_paths))
    
    ln_max = np.maximum.accumulate(ln_mx_paths, axis=1)

    St_final = np.exp(ln_St_paths[:, -1])
    St_max_all = np.exp(ln_max[:, -1])
    
    payoffs = payoff_func(St_final, St_max_all)
    price_vectors = payoffs.reshape(-1, iterations).mean(axis=1)
 
    discounted = np.exp(-r * (T - t)) * price_vectors
    mu = np.mean(discounted)
    std = np.std(discounted)
    end = time()
    if output:   
      print("--------- monde carlo-----------")
      print(f"mean: {mu}  std: {std}")
      print(f"Confidence Interval: [{mu - 2 * std}, {mu + 2 * std}]")
      print(f"taken time: {end - start} seconds")

def Cheuk_and_Vorst(option_type, S0, T, r, sigma, N, q, t, payoff_fuct, output = False):
    start = time()
    V_tree = np.zeros((N + 1, N + 1))  
    dt = (T-t) / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    mu = np.exp((r - q) * dt)
    q = (mu * u - 1) / (mu * (u - d))
    discount = np.exp(-r * dt)
    u_vector = [1]

    for i in range(1, N + 1):
        u_vector.append(u_vector[-1] * u)
        V_tree[i, N] = max(u_vector[-1] - 1, 0) 

    for t in range(N - 1, -1, -1):
        for d in range(t + 1):
            V_tree[d, t] = (q * V_tree[max(0, d - 1), t + 1] + (1 - q) * V_tree[d + 1, t + 1]) * discount * mu
            if option_type == "A":
                V_tree[d, t] = max(V_tree[d, t], u_vector[d] - 1)

    end = time()
    if output: 
       print("-------Cheuk_and_Vorst method---------")  
       print(f"value: {V_tree[0][0] * S0} taken time: {end - start} seconds")


def main():
    # binomial_lookback_option_price(option_type = "A", 
    #                                S = 50, K = 50, T = 0.25, r = 0.1, t = 0,
    #                                sigma = 0.4, N = 100, q = 0, payoff_fuct = payoff_put, 
    #                                S_mx_t = 50, output = True, optimize = True)
    # binomial_lookback_option_price(option_type = "E", 
    #                                S = 50, K = 50, T = 0.25, r = 0.1, t = 0,
    #                                sigma = 0.4, N = 100, q = 0, payoff_fuct = payoff_put, 
    #                                S_mx_t = 60, output = True, optimize = True)
    binomial_lookback_option_price(option_type = "A", 
                                   S = 50, K = 50, T = 0.25, r = 0.1, t = 0,
                                   sigma = 0.4, N = 100, q = 0, payoff_fuct = payoff_put, 
                                   S_mx_t = 70, output = True, optimize = True)
    # binomial_lookback_option_price(option_type = "A", 
    #                                S = 50, K = 50, T = 0.25, r = 0.1, t = 0,
    #                                sigma = 0.4, N = 100, q = 0, payoff_fuct = payoff_put, 
    #                                S_mx_t = 50, output = True, optimize = True)
    # binomial_lookback_option_price(option_type = "A", 
    #                                S = 50, K = 50, T = 0.25, r = 0.1, t = 0,
    #                                sigma = 0.4, N = 100, q = 0, payoff_fuct = payoff_put, 
    #                                S_mx_t = 60, output = True, optimize = True)
    # binomial_lookback_option_price(option_type = "A", 
    #                                S = 50, K = 50, T = 0.25, r = 0.1, t = 0,
    #                                sigma = 0.4, N = 100, q = 0, payoff_fuct = payoff_put, 
    #                                S_mx_t = 70, output = True, optimize = True)
    # monte_carlo_lookback(St = 50, T = 0.25, r = 0.1, sigma = 0.4, N = 100, q = 0, payoff_func = payoff_put, St_max = 50, output = True, sample_size = 20, iterations = 10000, t = 0)
    # monte_carlo_lookback(St = 50, T = 0.25, r = 0.1, sigma = 0.4, N = 100, q = 0, payoff_func = payoff_put, St_max = 60, output = True, sample_size = 20, iterations = 10000, t = 0)
    # monte_carlo_lookback(St = 50, T = 0.25, r = 0.1, sigma = 0.4, N = 100, q = 0, payoff_func = payoff_put, St_max = 70, output = True, sample_size = 20, iterations = 10000, t = 0)
    # Cheuk_and_Vorst(option_type = "", S0 = 50, T = 0.25, r = 0.1, 
    #                 sigma = 0.4, N = 1000, q = 0, payoff_fuct = payoff_put, output = True, t = 0)

if __name__ == "__main__":
    main()
