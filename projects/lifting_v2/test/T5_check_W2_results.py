import numpy as np

#  ±   ±   ±   ±   ± 4.68 9.481 ± 2.439
# 24.769 ± 12.06 14.76 ± 6.628 8.44 ± 1.524 6.175 ± 0.559 4.66 ± 0.508 3.984 ± 0.538
# 6.58 ± 0.589 5.731 ± 0.489 4.129 ± 0.378 3.049 ± 0.33 2.318 ± 0.339 1.998 ± 0.387

r1_mean_W2 = np.array([108.68, 101.495, 74.179, 34.403, 12.283])
r1_std_W2 = np.array([7.196, 8.781, 13.95, 14.731])

r2_mean_W2 = np.array([59.715, 39.825, 9.066, 5.377, 3.942, 3.354])
r2_std_W2 = np.array([15.039, 14.946, 3.082, 0.54, 0.523, 0.669])

r3_mean_W2 = np.array([5.805, 4.921, 3.451, 2.528, 1.923, 1.66])
r3_std_W2 = np.array([0.533, 0.424, 0.32, 0.305, 0.362, 0.46])

etas = np.array([26 / 100, 30 / 100, 40 / 100, 50 / 100, 60 / 100, 66 / 100])

X1 = 14761
X2 = 114564
X3 = 860069

print("Check scaling between different size sets")
print((X3 / X2) ** (-(1 + etas) / 5))
print(r3_mean_W2 / r2_mean_W2)

print("Check scaling within a sampling set")
print("X2")
print(X2 ** ((etas[0:-1] - etas[1:]) / 5))
print(r2_mean_W2[1:] / r2_mean_W2[0:-1])
print("X3")
print(X3 ** ((etas[0:-1] - etas[1:]) / 5))
print(r3_mean_W2[1:] / r3_mean_W2[0:-1])

# r2_mean_W2 = np.array([59.715, 39.825, 9.066, 5.377, 3.942, 3.354])
# r2_std_W2 = np.array([15.039, 14.946, 3.082, 0.54, 0.523, 0.669])
#
# r3_mean_W2 = np.array([5.805, 4.921, 3.451, 2.528, 1.923, 1.66])
# r3_std_W2 = np.array([0.533, 0.424, 0.32, 0.305, 0.362, 0.46])

# TODO also check J!
eta_range = etas

r1_mean_J = [760.77, 517.52, 173.46, 47.068, 16.167, 9.889]
r1_std_J = [92.25, 68.78, 32.22, 10.762, 2.953, 1.863]

r2_mean_J = [253.32, 151.67, 55.684, 22.732, 9.655, 5.948]
r2_std_J = [51.7, 25.39, 7.449, 3.356, 1.916, 1.413]

r3_mean_J = [204.95,  135.53, 49.646, 19.032, 7.847, 4.721]
r3_std_J = [25.76, 17.07, 6.585, 3.031, 1.771, 1.339]

r1_scaling_theory = X1 ** (3*(eta_range[-1] - np.array(eta_range[0:-1]))/5)
r1_scaling_practice = np.array(r1_mean_J)[0:-1]/np.array(r1_mean_J)[-1]

print(np.round(r1_scaling_theory,3))
print(np.round(r1_scaling_practice,3))

r2_scaling_theory = X2 ** (3*(eta_range[-1] - np.array(eta_range[0:-1]))/5)
r2_scaling_practice = np.array(r2_mean_J)[0:-1]/np.array(r2_mean_J)[-1]

print(np.round(r2_scaling_theory,3))
print(np.round(r2_scaling_practice,3))

r3_scaling_theory = X3 ** (3*(eta_range[-1] - np.array(eta_range[0:-1]))/5)
r3_scaling_practice = np.array(r3_mean_J)[0:-1]/np.array(r3_mean_J)[-1]

print(np.round(r3_scaling_theory,3))
print(np.round(r3_scaling_practice,3))
