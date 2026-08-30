import numpy as np

def iterative_polynomial_fitting_2(x_input, poly_order = 5, niter = 100, mode = "new"):
    x_new = np.copy(x_input)
    x_origin = np.copy(x_input)

    spacing = np.linspace(1, x_input.shape[-1], x_input.shape[-1])
    for i in range(niter):
        if mode == "new":
            coeffs = np.polyfit(spacing, np.transpose(x_new), poly_order) # coeffs is (poly_order + 1, num_samples)
            baseline_fitted = np.polynomial.polynomial.polyval(spacing, coeffs[-1::-1]) # baseline_fitted is (num_samples, seq_len)
            residuals = x_new - baseline_fitted
            stdev = np.std(residuals, axis = 1)
            peaks = np.where(x_new > baseline_fitted + np.expand_dims(stdev, axis = 1))
            # print(peaks)
            x_new[peaks] = (baseline_fitted + np.expand_dims(stdev, axis = 1))[peaks]

        else:
            coeffs = np.polyfit(spacing, np.transpose(x_new), poly_order) # coeffs is (poly_order + 1, num_samples)
            baseline_fitted = np.polynomial.polynomial.polyval(spacing, coeffs[-1::-1]) # baseline_fitted is (num_samples, seq_len)
            x_new = np.minimum(baseline_fitted, x_origin)

    del x_new
    del x_origin
    # gc.collect()

    # plt.figure(figsize=(10, 6))
    # plt.plot(shifts, x_input[0])
    # plt.plot(shifts, baseline_fitted[0])
    # plt.plot(shifts, x_input[0] - baseline_fitted[0])
    # plt.savefig("test_ipf.png")

    return x_input - baseline_fitted