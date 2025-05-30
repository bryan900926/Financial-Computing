import numpy as np
from numba import njit  
from time import time

def payoff_call(S_vector: int, K: int)-> int:
    return np.maximum(S_vector - K, 0)

class S_AVG_node:
    def __init__(self, mn, mx, M):
        self.s_avg = np.linspace(mn, mx, num = M + 1)
        self.option_value = np.zeros(M + 1)
        self.min = mn
        self.gap = (mx - mn) / M

    def interpolate_option_value(self, avg_price):
        if not self.gap:
            return self.option_value[0]
    
        idx = min(max(int((avg_price - self.min) / self.gap), 0), len(self.s_avg) - 2)
    
        lower_avg = self.s_avg[idx]
        upper_avg = self.s_avg[idx + 1]
        lower_option = self.option_value[idx]
        upper_option = self.option_value[idx + 1]
    
        denom = upper_avg - lower_avg
        if denom == 0:
            return lower_option
    
        w = (avg_price - lower_avg) / denom
        return (1 - w) * lower_option + w * upper_option
    
def asian_option_CRR(option_type, St, K, T, t, r, sigma, N, q, payoff_fuct, S_AVG_t, m, output = False):
    start = time()
    dt = (T - t) / N
    cnt_t = int(t / dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    stock_tree = np.zeros((N + 1, N + 1))
    stock_AVG_tree = [[0] * (N + 1) for _ in range(N + 1)]
    stock_tree[0, 0] = St

    u_vector, d_vector = [1], [1]
    for _ in range(N):
        u_vector.append(u_vector[-1] * u)
        d_vector.append(d_vector[-1] * d)

    for i in range(N):
        stock_tree[:i + 1, i + 1] = stock_tree[:i + 1, i] * u
        stock_tree[i + 1, i + 1] = stock_tree[i, i] * d

    for i in range(N + 1):
        for j in range(i + 1):
            up = i - j
            A_mx = (S_AVG_t * (cnt_t + 1) + St * u * (1 - u_vector[up]) / (1 - u) + St * u_vector[up] * d * (1 - d_vector[j]) / (1 - d)) / (i + 1 + cnt_t) 
            A_mn = (S_AVG_t * (cnt_t + 1) + St * d_vector[j] * u * (1 - u_vector[up]) / (1 - u) + St * d * (1 - d_vector[j]) / (1 - d)) / (i + 1 + cnt_t)
            stock_AVG_tree[j][i] = S_AVG_node(mx = A_mx, mn = A_mn, M = m)   

    for j in range(N + 1): 
        avg_price = stock_AVG_tree[j][N].s_avg
        option = stock_AVG_tree[j][N].option_value 
        for i in range(len(avg_price)):
            option[i] = payoff_fuct(avg_price[i], K)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1): 
            avg_price = stock_AVG_tree[j][i].s_avg
            option = stock_AVG_tree[j][i].option_value 
            for k in range(len(avg_price)):
                A = avg_price[k]
                A_u = ((i + 1 + cnt_t) * A + St * u_vector[i + 1 - j] * d_vector[j]) / (i + 2 + cnt_t)
                A_d = ((i + 1 + cnt_t) * A + St * u_vector[i - j] * d_vector[j + 1]) / (i + 2 + cnt_t)  
                option_u = stock_AVG_tree[j][i + 1].interpolate_option_value(A_u)
                option_d = stock_AVG_tree[j + 1][i + 1].interpolate_option_value(A_d)
                option[k] = (p * option_u + (1 - p) * option_d)
                if option_type == 'A':
                    option[k] = max(option[k], payoff_fuct(avg_price[k], K))
                option[k] *= discount
    end = time()
    print(stock_AVG_tree[0][0].option_value[0], end - start)
    return stock_AVG_tree[0][0].option_value[0]

def monte_carlo_lookback_asian(N, S_AVG_t, St, r, t, q, T, K, sigma, sample_size, iterations, payoff_func, output = False) -> str:
    start = time()
    dt = (T - t) / N
    cnt_t = int(t / dt)
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    total_paths = iterations * sample_size
    Z = np.random.randn(total_paths, N)

    ln_St_paths = np.log(St) + np.cumsum(drift + diffusion * Z, axis=1)
    St_paths = np.exp(ln_St_paths)
    St_mean = (S_AVG_t * (cnt_t + 1) + np.sum(St_paths, axis = 1)) / (cnt_t + N + 1)
    payoff_vector = payoff_func(St_mean, K) * np.exp(-r * T)
    price_vector = payoff_vector.reshape(-1, iterations).mean(axis=1)
    mu, std = np.mean(price_vector), np.std(price_vector)

    end = time()
    if output:   
      print("--------- monde carlo-----------")
      print(f"mean: {mu}  std: {std}")
      print(f"Confidence Interval: [{mu - 2 * std}, {mu + 2 * std}]")
      print(f"taken time: {end - start} seconds")

def main():
    # asian_option_CRR(option_type = "", St = 50, K = 40, 
    #                 T = 0.5, t = 0, r = 0.2, sigma = 0.2, N = 150, q = 0, payoff_fuct = payoff_call, S_AVG_t = 50, m = 250)
    monte_carlo_lookback_asian(St = 50, K = 40, 
                    T = 0.5, t = 0, r = 0.2, sigma = 0.2, N = 150, q = 0,
                    payoff_func = payoff_call, S_AVG_t = 50, sample_size = 20, iterations = 10000, output = True)
if __name__ == "__main__":
    main()
