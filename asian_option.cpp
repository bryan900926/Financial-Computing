#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using Clock = std::chrono::high_resolution_clock;

double payoff_call(double S, double K) {
    return std::max(S - K, 0.0);
}

class S_AVG_node {
public:
    std::vector<double> s_avg;
    std::vector<double> option_value;
    double min;
    double gap;
    
    S_AVG_node() : s_avg(), option_value(), min(0), gap(0) {}

    S_AVG_node(double mn, double mx, int M, bool if_linear) {
        min = mn;
        gap = (mx - mn) / M;
        s_avg.resize(M + 1);
        option_value.resize(M + 1, 0.0);
        s_avg[0] = mn;
        for (int k = 1; k <= M; k++) {
            if (if_linear)  s_avg[k] = s_avg[k - 1] + gap;
            else{
                double w = static_cast<double>(k) / M;
                s_avg[k] = std::exp(w * std::log(mx) + (1 - w) * std::log(mn));
            }
        }
    }

    double interpolate_option_value(double avg_price, int method) {
        if (gap == 0) return option_value[0];
        int idx = 0;
        if (method == 0){
            idx = lower_bound(s_avg.begin(), s_avg.end(), avg_price) - s_avg.begin() - 1;
        }
        else if (method == 1){
            idx = static_cast<int>((avg_price - min) / gap);
        }
        else{
            for (int i = 0; i < s_avg.size(); i++){
                if (s_avg[i] >= avg_price){
                    idx = i - 1;
                    break;
                }
            }
        }
        
        idx = std::max(0, std::min(idx, static_cast<int>(s_avg.size()) - 2));

        double lower_avg = s_avg[idx];
        double upper_avg = s_avg[idx + 1];
        double lower_option = option_value[idx];
        double upper_option = option_value[idx + 1];

        double denom = upper_avg - lower_avg;
        if (denom == 0) return lower_option;

        double w = (avg_price - lower_avg) / denom;
        return (1 - w) * lower_option + w * upper_option;
    }
};

double asian_option_CRR(
    std::string option_type, double St, double K, double T, double t, double r, double sigma,
    int N, double q, double (*payoff_func)(double, double), double S_AVG_t, int m, bool output = false, int method = 0, bool if_linear = true
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    double dt = (T - t) / N;
    int cnt_t = static_cast<int>(t / dt);
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp((r - q) * dt) - d) / (u - d);
    double discount = std::exp(-r * dt);

    std::vector<std::vector<double>> stock_tree(N + 1, std::vector<double>(N + 1, 0.0));
    std::vector<std::vector<S_AVG_node>> stock_AVG_tree(N + 1);
    for (int i = 0; i <= N; ++i) {
        stock_AVG_tree[i].resize(i + 1);
    }

    stock_tree[0][0] = St;

    std::vector<double> u_vector = {1.0}, d_vector = {1.0};
    for (int i = 1; i <= N; i++) {
        u_vector.push_back(u_vector.back() * u);
        d_vector.push_back(d_vector.back() * d);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; ++j) {
            stock_tree[j][i + 1] = stock_tree[j][i] * u;
        }
        stock_tree[i + 1][i + 1] = stock_tree[i][i] * d;
    }

    for (int i = 0; i <= N; i++) {
        stock_AVG_tree[i].resize(i + 1);
        for (int j = 0; j <= i; j++) {
            int up = i - j;
            double A_mx = (S_AVG_t * (cnt_t + 1) +
                St * u * (1 - u_vector[up]) / (1 - u) +
                St * u_vector[up] * d * (1 - d_vector[j]) / (1 - d)) / (i + 1 + cnt_t);

            double A_mn = (S_AVG_t * (cnt_t + 1) +
                St * d_vector[j] * u * (1 - u_vector[up]) / (1 - u) +
                St * d * (1 - d_vector[j]) / (1 - d)) / (i + 1 + cnt_t);

            stock_AVG_tree[i][j] = S_AVG_node(A_mn, A_mx, m, if_linear);
        }
    }
    for (int j = 0; j <= N; ++j) {
        auto& avg_price = stock_AVG_tree[N][j].s_avg;
        auto& option = stock_AVG_tree[N][j].option_value;
        for (size_t i = 0; i < avg_price.size(); i++) {
            option[i] = payoff_func(avg_price[i], K);
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            auto& avg_price = stock_AVG_tree[i][j].s_avg;
            auto& option = stock_AVG_tree[i][j].option_value;

            for (size_t k = 0; k < avg_price.size(); ++k) {
                double A = avg_price[k];
                double A_u = ((i + 1 + cnt_t) * A + St * u_vector[i + 1 - j] * d_vector[j]) / (i + 2 + cnt_t);
                double A_d = ((i + 1 + cnt_t) * A + St * u_vector[i - j] * d_vector[j + 1]) / (i + 2 + cnt_t);

                double option_u = stock_AVG_tree[i + 1][j].interpolate_option_value(A_u, method);
                double option_d = stock_AVG_tree[i + 1][j + 1].interpolate_option_value(A_d, method);

                option[k] = (p * option_u + (1 - p) * option_d) * discount;
                if (option_type == "A") {
                    option[k] = std::max(option[k], payoff_func(avg_price[k], K));
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double result = stock_AVG_tree[0][0].option_value[0];
    if (output) {
        std::cout << "Option Value: " << result << std::endl;
        std::cout << "Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s\n";
    }
    return result;
}


double monte_carlo_lookback_asian(
    int N, double S_AVG_t, double St, double r, double t, double q, double T, double K, double sigma,
    int sample_size, int iterations,
    std::function<double(double, double)> payoff_func,
    bool output = false
) {
    auto start_time = Clock::now();

    double dt = (T - t) / N;
    int cnt_t = static_cast<int>(t / dt);
    double drift = (r - q - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * sqrt(dt);


    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> norm(0.0, 1.0);

    std::vector<double> price_vector(iterations, 0.0);

    for (int iter = 0; iter < iterations; ++iter) {
        double sum_payoff = 0.0;

        for (int s = 0; s < sample_size; ++s) {
            std::vector<double> log_St(N + 1, log(St));

            for (int i = 1; i <= N; ++i) {
                log_St[i] = log_St[i - 1] + drift + diffusion * norm(gen);
            }

            std::vector<double> St_path(N);
            transform(log_St.begin() + 1, log_St.end(), St_path.begin(), [](double x) { return exp(x); });
            
            double St_mean = (S_AVG_t * (cnt_t + 1) + accumulate(St_path.begin(), St_path.end(), 0.0)) / (cnt_t + 1 + N);
            double payoff = payoff_func(St_mean, K) * exp(-r * T);
            sum_payoff += payoff;
        }

        price_vector[iter] = sum_payoff / sample_size;
    }

    double mu = accumulate(price_vector.begin(), price_vector.end(), 0.0) / iterations;
    double sq_sum = inner_product(price_vector.begin(), price_vector.end(), price_vector.begin(), 0.0);
    double std = sqrt(sq_sum / iterations - mu * mu);

    auto end_time = Clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    if (output) {
        std::cout << "--------- monte carlo -----------\n";
        std::cout << "mean: " << mu << "  std: " << std << "\n";
        std::cout << "Confidence Interval: [" << mu - 2 * std << ", " << mu + 2 * std << "]\n";
        std::cout << "taken time: " << duration << " seconds\n";
    }

    return mu;
}

void plot_convergence(int M, int start_m){
    std::vector<int> m1;
    std::vector<int> m2;
    std::vector<double> option_value1;
    std::vector<double> option_value2;
    for (int i = start_m; i <= M; i+=50){
        m1.push_back(i);
        m2.push_back(i);
        double result1 = asian_option_CRR(
        "",      // option_type
        50.0,    // St
        40.0,    // K
        0.5,     // T
        0.0,     // t
        0.2,     // r
        0.2,     // sigma
        100,     // N
        0.0,     // q
        payoff_call,  // payoff function
        50.0,    // S_AVG_t
        i,    // m
        false,
        0,
        true
        );

        double result2 = asian_option_CRR(
        "",      // option_type
        50.0,    // St
        40.0,    // K
        0.5,     // T
        0.0,     // t
        0.2,     // r
        0.2,     // sigma
        100,     // N
        0.0,     // q
        payoff_call,  // payoff function
        50.0,    // S_AVG_t
        i,    // m
        false, // output res
        0, // method
        false // if_linear
        );
        option_value1.push_back(result1);
        option_value2.push_back(result2);
    } 

    plt::plot(m1, option_value1, {{"label", "linear Method"}});
    plt::plot(m2, option_value2, {{"label", "logarithmical Method"}});
    plt::title("Convergence Plot");
    plt::xlabel("Number of Grid Points (m)");
    plt::ylabel("Option Value");
    plt::legend();
    plt::show();
}

// int N, double S_AVG_t, double St, double r, double t, double q, double T, double K, double sigma,
// int sample_size, int iterations,
// std::function<double(double, double)> payoff_func,
// bool output = false
int main() {
    // double result = asian_option_CRR(
    //     "",      // option_type
    //     50.0,    // St
    //     40.0,    // K
    //     0.5,     // T
    //     0.0,     // t
    //     0.2,     // r
    //     0.2,     // sigma
    //     100,     // N
    //     0.0,     // q
    //     payoff_call,  // payoff function
    //     50.0,    // S_AVG_t
    //     30,    // m
    //     true,  // output the res
    //     0,  // method
    //     false // if_linear
    // );
    // auto res = monte_carlo_lookback_asian(
    //     100, // N
    //     50, // S_AVG_t
    //     50, // St
    //     0.2, // r
    //     0.0, // t
    //     0, //q
    //     0.5, //T
    //     40, //K
    //     0.2, //sigma
    //     10000, //sample size
    //     20, //iterations
    //     payoff_call,
    //     true
    // );
    plot_convergence(600, 50);
    // auto s = S_AVG_node(40, 50, 100, false);
    // for (auto e: s.s_avg){
    //     std::cout << e << std::endl;
    // }
    return 0; 
}
//  g++ hw5.cpp -IC:/Users/bryan/miniconda3/include -IC:/Users/bryan/AppData/Roaming/Python/Python312/site-packages/numpy/core/include -IC:/c_library -LC:/Users/bryan/miniconda3/libs -lpython312 -o hw5.exe
