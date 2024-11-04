import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os

def piecewise_curve_function_v1(x, m1, d1, c1, d2, z):
    m2 = (m1 * d1) / (d2 * np.exp(d1 * z))
    c2 = (m1 * np.exp(-d1 * z)) + c1 - m2
    return np.piecewise(x, [x <= z, x > z], [lambda x: m1 * np.exp(-d1 * x) + c1, lambda x: m2 * np.exp(-d2 * (x - z)) + c2])

def piecewise_curve_function_v2(x, m1, d1, c1, m2, d2, z):
    c2 = (m1 * np.exp(-d1 * z)) + c1 - m2
    return np.piecewise(x, [x <= z, x > z], [lambda x: m1 * np.exp(-d1 * x) + c1, lambda x: m2 * np.exp(-d2 * (x - z)) + c2])

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    x_values, v_values = data[:, 0], data[:, 1]
    return x_values, v_values

def fit_curve_v1(x_values, v_values):
    initial_guess = [-8, 0.0005, 170, 1/100, (0.5 * np.mean(x_values))]  # Initial guess for m1, d1, c1, d2, z

    # Set bounds for the parameters
    bounds = ([-400, 1/10000, 110, 5/1000, 0], [400, 1, 1000, 1, np.mean(x_values)])

    params, covariance = curve_fit(piecewise_curve_function_v1, x_values, v_values, p0=initial_guess, bounds=bounds)
    return params

def fit_curve_v2(x_values, v_values):
    thold_secondcurve = 2
    d2_initial = 0.005  # Initial guess for d2
    initial_guess = [-8, 0.000005, 170, -abs(thold_secondcurve * np.exp(100 * d2_initial) - thold_secondcurve), 5 / 1000,
                     (0.5 * np.mean(x_values))]  # Initial guess for m1, d1, c1, m2, d2, z

    # Set initial bounds for m2
    # Adjust the lower bound to be a bit lower to capture smaller deviations
    bounds = ([-400, 0, 110, -400, 5 / 1000, 0],
              [400, 1, 1000, max(-399, -abs(thold_secondcurve * np.exp(100 * d2_initial) - thold_secondcurve)), 5 / 100, np.mean(x_values)])
    #bounds = ([-400, 0, 110, -10 * abs(thold_secondcurve * np.exp(100 * d2_initial) - thold_secondcurve), 0, 0],
              #[400, 1, 1000, -abs(thold_secondcurve * np.exp(100 * d2_initial) - thold_secondcurve), 1, np.mean(x_values)])

    params, covariance = curve_fit(piecewise_curve_function_v2, x_values, v_values, p0=initial_guess, bounds=bounds)
    m1, d1, c1, m2, d2, z = params

    print(f"{[*params]}")
    print(f"d2={d2}")

    # Update initial guess and bounds for m2 using d2 from the first optimization
    initial_guess = [-8, 0.000005, 170, max(-399,-abs(thold_secondcurve * np.exp(100 * d2) - thold_secondcurve)), 5 / 1000,
                     (0.5 * np.mean(x_values))]  # Initial guess for m1, d1, c1, m2, d2, z
    bounds = ([-400, 0, 110, -400, 5 / 1000, 0],
              [400, 1, 1000, max(-thold_secondcurve, -abs(thold_secondcurve * np.exp(100 * d2) - thold_secondcurve)), 5 / 100, np.mean(x_values)])

    # Perform the optimization with updated initial guess and bounds
    params, covariance = curve_fit(piecewise_curve_function_v2, x_values, v_values, p0=initial_guess, bounds=bounds)
    m1, d1, c1, m2, d2, z = params
    print(f"{[*params]}")
    print(f"d2={d2}")

    return params

def calculate_ratio_v1(x_values, d1, d2, z):
    ratio = abs(np.exp(-z * (d1 + d2) - d2 * np.max(x_values)) * 100)
    return ratio

def calculate_ratio_v2(x_values, m1, d1, m2, d2, z):
    ratio = abs(m2 * d2 / (m1 * d1 * np.exp(d2 * (np.max(x_values) - z))) * 100)
    return ratio

def calculate_standard_deviation_v1(x_values, v_values, fitted_params):
    fitted_curve_values = piecewise_curve_function_v1(x_values, *fitted_params)
    differences_v1 = v_values - fitted_curve_values
    std_v1 = np.std(differences_v1)
    return std_v1

def calculate_standard_deviation_v2(x_values, v_values, fitted_params):
    fitted_curve_values = piecewise_curve_function_v2(x_values, *fitted_params)
    differences_v2 = v_values - fitted_curve_values
    std_v2 = np.std(differences_v2)
    return std_v2

def plot_data_and_curve(x_values, v_values, fitted_params_v1, fitted_params_v2, who_won):
    sorted_indices = np.argsort(x_values)
    x_values_sorted = x_values[sorted_indices]
    v_values_sorted = v_values[sorted_indices]

    sorted_v_values = np.sort(v_values_sorted)
    lower_limit = (np.percentile(sorted_v_values, 0.5) - 1)
    upper_limit = (np.percentile(sorted_v_values, 99.5) + 1)

    # Plot the original data and the fitted piecewise curves
    plt.scatter(x_values_sorted, v_values_sorted, label='Actual Data')

    # Plot fitted curves for both versions
    if who_won == 1:
        plt.plot(x_values_sorted, piecewise_curve_function_v1(x_values_sorted, *fitted_params_v1), label='Fitted Piecewise Curve v1 [won]', color='green')
        plt.plot(x_values_sorted, piecewise_curve_function_v2(x_values_sorted, *fitted_params_v2), label='Fitted Piecewise Curve v2', color='red')
    else:
        plt.plot(x_values_sorted, piecewise_curve_function_v1(x_values_sorted, *fitted_params_v1), label='Fitted Piecewise Curve v1', color='red')
        plt.plot(x_values_sorted, piecewise_curve_function_v2(x_values_sorted, *fitted_params_v2), label='Fitted Piecewise Curve v2 [won]', color='green')

    plt.ylim(lower_limit, upper_limit)

    plt.legend()
    plt.xlabel('x values')
    plt.ylabel('v values')
    plt.title('Piecewise Curve Fitting')

    plt.show()

def play_completion_sound():
    os.system("afplay /System/Library/Sounds/Glass.aiff")

def main(file_path):
    consecutive_below_threshold = 0
    start_time = time.time()

    while True:
        x_values, v_values = load_data(file_path)

        try:
            # Check if at least 4 minutes have passed
            elapsed_time = time.time() - start_time
            if elapsed_time >= 0:  # 4 minutes in seconds -> 240 but use 0 for faster testing
                # Attempt to fit the piecewise curves
                params_v1 = fit_curve_v1(x_values, v_values)
                params_v2 = fit_curve_v2(x_values, v_values)

                std_v1 = calculate_standard_deviation_v1(x_values, v_values, params_v1)
                std_v2 = calculate_standard_deviation_v2(x_values, v_values, params_v2)
                #print(f"std1={std_v1}, std2={std_v2}")

                # Check if both functions produced a solution within their bounds
                if std_v1 <= std_v2:
                    print(f"{[*params_v2]}")
                    ratio = calculate_ratio_v1(x_values, params_v1[1], params_v1[3], params_v1[4])
                    who_won = 1
                else:
                    ratio = calculate_ratio_v2(x_values, params_v2[0], params_v2[1], params_v2[3], params_v2[4], params_v2[5])
                    who_won = 2
                print(f"Ratio m2-End/m1-Start: {round(ratio, 2)}%")

                # Check if the ratio is below 5.5% for two consecutive runs
                if ratio < 5.5:
                    consecutive_below_threshold += 1
                    if consecutive_below_threshold >= 6:  # 6 seconds threshold
                        break  # Exit the while loop if the threshold is reached
                else:
                    consecutive_below_threshold = 0

                # Plot the data and fitted curve of the function with lower standard deviation
                #plot_data_and_curve(x_values, v_values, params_v1, params_v2, who_won)

                # Play a sound to indicate completion
                play_completion_sound()

            else:
                print(f"Waiting for at least 4 minutes of data... Elapsed time: {round(elapsed_time, 2)} seconds")

        except Exception as e:
            print("Error: Can't calculate yet. Check your data or initial guesses.")
            print(f"Error details: {e}")

        #time.sleep(1)  # Add a delay before the next iteration to avoid excessive CPU usage

    if consecutive_below_threshold >= 6:
        # Plot the data and fitted curve of the function with lower standard deviation
        plot_data_and_curve(x_values, v_values, params_v1, params_v2, who_won)

        # Play a sound to indicate completion
        play_completion_sound()


if __name__ == "__main__":
    file_path = r"/Users/manuelkalozi/Desktop/Clear1_24.08.24.txt"
    main(file_path)
