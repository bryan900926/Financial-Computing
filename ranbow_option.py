import numpy as np
import pandas as pd

def cholesky(C):
    n = C.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = (C[i, i] - np.sum(A[:i, i] ** 2)) ** 0.5
        for j in range(i + 1, n):
            A[i, j] = (C[i][j] - np.sum(A[:i, i] * A[:i, j])) / A[i, i]
    return A

def ranbow_payoff(price_vector, K):
    return max(np.max(price_vector) - K,  0)

def monte_carlo(S0_vector, r, q_vector,  T, sigma_vector, rho_matrix,
                K: int, sample_size: int, iterations: int, payoff_fuct) -> str:

    n = sigma_vector.shape[0]
    mean, variance = np.log(S0_vector) + (r - q_vector - 0.5 * sigma_vector ** 2) * T, np.square(sigma_vector) * T
    cov_matrix = np.zeros((n, n))
    mean_mat = np.tile(mean, (sample_size, 1))

    for i in range(n):
        for j in range(i, n):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_matrix[i, j] * (variance[i] * variance[j]) ** 0.5 

    price_vector = np.zeros(iterations)
    price_vector_reduct = np.zeros(iterations)
    price_vector_wang = np.zeros(iterations)
    A = cholesky(cov_matrix)

    for j in range(iterations):

        Z = np.random.randn(sample_size, n) 
        half_Z = Z[:sample_size // 2, :] 
        Z_reduct = np.vstack((half_Z, -half_Z))     

        for i in range(n):
            Z_reduct[:,i] /= np.std(Z_reduct[:,i])

        A_star = cholesky(np.cov(Z_reduct, rowvar=False))
        Z_wang = (Z_reduct) @ np.linalg.inv(A_star)

        St_mat = np.exp(Z @ A + mean_mat)
        St_mat_reduct = np.exp(Z_reduct @ A + mean_mat)
        St_mat_wang = np.exp(Z_wang @ A + mean_mat)

        price_vector[j] = np.mean(np.apply_along_axis(payoff_fuct, 1, St_mat, K)) * np.exp(-r * T)
        price_vector_reduct[j] = np.mean(np.apply_along_axis(payoff_fuct, 1, St_mat_reduct, K)) * np.exp(-r * T)
        price_vector_wang[j] = np.mean(np.apply_along_axis(payoff_fuct, 1, St_mat_wang, K)) * np.exp(-r * T)

    mu, std = np.mean(price_vector), np.std(price_vector)
    mu_reduct, std_reduct = np.mean(price_vector_reduct), np.std(price_vector_reduct)
    mu_wang, std_wang = np.mean(price_vector_wang), np.std(price_vector_wang)

    print(f"Method 1 mean: {mu}  std: {std}")
    print(f"Method 1 Confidence Interval: [{mu - 2 * std}, {mu + 2 * std}]")
    print(f"Method 2 mean: {mu_reduct}  std: {std_reduct}")
    print(f"Method 2 Confidence Interval: [{mu_reduct - 2 * std_reduct}, {mu_reduct + 2 * std_reduct}]")  
    print(f"Method 3 mean: {mu_wang}  std: {std_wang}")
    print(f"Method 3 Confidence Interval: [{mu_wang - 2 * std_wang}, {mu_wang + 2 * std_wang}]")  

# monte_carlo(S0_vector = np.array([95, 95]), r = 0.1, q_vector = np.array([0.05, 0.05]), T = 0.5, 
#             sigma_vector = np.array([0.5, 0.5]), rho_matrix = np.array([[ 1, 1],
#                                                                         [1, 1]]),
#             iterations = 20, sample_size = 10000, K = 100, payoff_fuct = ranbow_payoff)

# monte_carlo(S0_vector = np.array([95, 95]), r = 0.1, q_vector = np.array([0.05, 0.05]), T = 0.5, 
#             sigma_vector = np.array([0.5, 0.5]), rho_matrix = np.array([[ 1, -1],
#                                                                         [-1,  1]]),
#             iterations = 20, sample_size = 10000, K = 100, payoff_fuct = ranbow_payoff)

monte_carlo(S0_vector = np.array([95, 95, 95, 95, 95]), r = 0.1, q_vector = np.array([0.05, 0.05, 0.05, 0.05, 0.05]), T = 0.5, 
            sigma_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5]), rho_matrix = np.array([
                                                                                [ 1, 0.5,  0.5, 0.5, 0.5],
                                                                                [0.5,  1, 0.5, 0.5, 0.5],
                                                                                [ 0.5, 0.5,  1, 0.5, 0.5],
                                                                                [ 0.5, 0.5,  0.5, 1, 0.5],
                                                                                [ 0.5, 0.5,  0.5, 0.5, 1],
                                                                                      ]),
            iterations = 20, sample_size = 10000, K = 100, payoff_fuct = ranbow_payoff)


# c = covariance_matrix([[1, 0.2, 0.8], [1, 2, 0.4], [1, 2, 8]])
# print(cholesky(c))
    
