import numpy as np

r2_mean_W2 = np.array([59.715, 39.825, 9.066, 5.377, 3.942, 3.354])
r2_std_W2 = np.array([15.039, 14.946, 3.082, 0.54, 0.523, 0.669])

r3_mean_W2 = np.array([5.805, 4.921, 3.451, 2.528, 1.923, 1.66])
r3_std_W2 = np.array([0.533, 0.424, 0.32, 0.305, 0.362, 0.46])

etas = np.array([26/100, 30/100, 40/100, 50/100, 60/100, 66/100])

X2 = 114564
X3 = 860069

print("Check scaling between different size sets")
print((X3/X2)**(-(1+etas)/5))
print(r3_mean_W2/r2_mean_W2)

print("Check scaling within a sampling set")
print("X2")
print(X2**((etas[0:-1] - etas[1:])/5))
print(r2_mean_W2[1:]/r2_mean_W2[0:-1])
print("X3")
print(X3**((etas[0:-1] - etas[1:])/5))
print(r3_mean_W2[1:]/r3_mean_W2[0:-1])


