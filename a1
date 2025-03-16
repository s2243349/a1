# %% [markdown]
# # Assignment A1: Computing and Numerics

# %% [markdown]
# ## Question 1: Area of a Triangle

# %% [markdown]
# ### 1.1 Heron's Formula
# Implement a function to calculate the area of a triangle using Heron's formula.

# %%
import numpy as np

def triangle_area_heron(a, b, c):
    # Calculate semi-perimeter
    s = (a + b + c) / 2
    # Calculate area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

# %% [markdown]
# ### 1.2 Kahan's Formula
# Implement a function to calculate the area of a triangle using Kahan's formula.

# %%
def triangle_area_kahan(a, b, c):
    # Sort sides in descending order
    sides = sorted([a, b, c], reverse=True)
    a, b, c = sides
    # Calculate area using Kahan's formula
    area = 0.25 * np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))
    return area

# %% [markdown]
# ### 1.3 Comparison and Discussion
# Compare the accuracy of Heron's and Kahan's formulas for a family of triangles.

# %%
import matplotlib.pyplot as plt

# Define the exact area for the given triangle family
def exact_area(epsilon):
    return np.sqrt(1 + epsilon**4)

# Generate a range of epsilon values
epsilons = np.logspace(-10, 0, 100)
heron_errors = []
kahan_errors = []

for eps in epsilons:
    a = 2 * eps
    b = c = np.sqrt(1 + eps**4) / eps
    exact = exact_area(eps)
    
    # Calculate areas using both formulas
    heron_area = triangle_area_heron(a, b, c)
    kahan_area = triangle_area_kahan(a, b, c)
    
    # Calculate relative errors
    heron_errors.append(abs(heron_area - exact) / exact)
    kahan_errors.append(abs(kahan_area - exact) / exact)

# Plot the results
plt.figure(figsize=(10, 6))
plt.loglog(epsilons, heron_errors, label="Heron's Formula Error")
plt.loglog(epsilons, kahan_errors, label="Kahan's Formula Error")
plt.xlabel("Epsilon")
plt.ylabel("Relative Error")
plt.title("Comparison of Heron's and Kahan's Formulas")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# **Discussion**:
# Heron's formula performs well for moderate values of \(\varepsilon\) but suffers from significant numerical instability as \(\varepsilon\) becomes very small. This is due to the subtraction of nearly equal numbers in the formula, leading to loss of precision. Kahan's formula, on the other hand, maintains high accuracy across all values of \(\varepsilon\) because it avoids such subtractions. This demonstrates the importance of numerical stability in geometric calculations.

# %% [markdown]
# ## Question 2: Numerical Linear Algebra

# %% [markdown]
# ### 2.1 Sequence Generation
# Implement a function to generate the sequence \(x_n = A^n x_0\).

# %%
def sequence_element(n):
    # Define the matrix A
    A = np.array([[0, 1], [1, 1]])
    
    # Define the initial vector x0
    x0 = np.array([1, 1])
    
    # Initialize x as x0
    x = x0
    
    # Loop to compute x_n = A^n x0
    for _ in range(n):
        x = A @ x  # Matrix-vector multiplication
    
    # Return the result as an integer array
    return x.astype(int)

# %% [markdown]
# ### 2.2 Eigenvalue Error Analysis
# Investigate the error \(e_n = \frac{\|A x_n - \alpha x_n\|}{\|x_n\|}\) for different values of \(n\).

# %%
# Compute the largest eigenvalue of A
A = np.array([[0, 1], [1, 1]])
eigenvalues, _ = np.linalg.eig(A)  # Get all eigenvalues
alpha = max(eigenvalues)  # Choose the largest eigenvalue

# Initialize lists to store n values and errors
n_values = range(1, 20)
errors = []

# Loop through each n value
for n in n_values:
    # Compute x_n using the sequence_element function
    x_n = sequence_element(n)
    
    # Compute A * x_n
    Ax_n = A @ x_n
    
    # Compute alpha * x_n
    alpha_x_n = alpha * x_n
    
    # Calculate the error e_n = ||A x_n - alpha x_n|| / ||x_n||
    error = np.linalg.norm(Ax_n - alpha_x_n) / np.linalg.norm(x_n)
    errors.append(error)

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogy(n_values, errors, marker='o')
plt.xlabel("n")
plt.ylabel("Error e_n")
plt.title("Error in Eigenvalue Approximation")
plt.grid()
plt.show()

# %% [markdown]
# **Discussion**:
# The error \(e_n\) decreases quickly as \(n\) increases, which shows that the sequence \(x_n\) is converging to the eigenvector of \(A\) corresponding to the largest eigenvalue. This is expected because the power method (which is what this sequence represents) is known to converge to the dominant eigenvector. The exponential decay of the error is typical for well-behaved matrices.

# %% [markdown]
# ## Question 3: Numerical Integration

# %% [markdown]
# ### 3.1 Quadrature Weights
# Implement a function to compute the weights for an interpolatory quadrature rule.

# %%
def interpolatory_quadrature_weights(x):
    # Get the number of points (N + 1)
    N = len(x) - 1
    
    # Initialize an array to store the weights
    w = np.zeros(N + 1)
    
    # Loop through each point to compute its weight
    for i in range(N + 1):
        # Start with the constant polynomial 1
        p = np.poly1d([1])
        
        # Construct the Lagrange basis polynomial
        for j in range(N + 1):
            if j != i:
                # Multiply by (x - x_j) / (x_i - x_j)
                p *= np.poly1d([1, -x[j]]) / (x[i] - x[j])
        
        # Integrate the polynomial from -1 to 1 to get the weight
        w[i] = np.polyint(p)(1) - np.polyint(p)(-1)
    
    return w  # Return the computed weights

# Test the function with midpoint, trapezoidal, and Simpson's rules
x_midpoint = np.array([0])  # Midpoint rule
x_trapezoidal = np.array([-1, 1])  # Trapezoidal rule
x_simpson = np.array([-1, 0, 1])  # Simpson's rule

print("Midpoint weights:", interpolatory_quadrature_weights(x_midpoint))
print("Trapezoidal weights:", interpolatory_quadrature_weights(x_trapezoidal))
print("Simpson's weights:", interpolatory_quadrature_weights(x_simpson))

# %% [markdown]
# ### 3.2 Comparison of Quadrature Rules
# Compare the accuracy of two quadrature rules for integrating a function.

# %%
# Define the function to integrate
def f(x):
    return 1 / (1 + (3 * x)**2)

# Define the two sets of quadrature points
def x0_points(N):
    # Uniformly spaced points
    return -1 + 2 * np.arange(N + 1) / N

def x1_points(N):
    # Chebyshev points
    return -np.cos(np.arange(N + 1) * np.pi / N)

# Compute the exact integral using scipy's quad function
from scipy.integrate import quad
exact_integral, _ = quad(f, -1, 1)

# Compare the two quadrature rules
N_values = range(1, 20)
errors_x0 = []
errors_x1 = []

for N in N_values:
    # Get the points and weights for both rules
    x0 = x0_points(N)
    x1 = x1_points(N)
    w0 = interpolatory_quadrature_weights(x0)
    w1 = interpolatory_quadrature_weights(x1)
    
    # Compute the approximate integrals
    integral_x0 = np.sum(w0 * f(x0))
    integral_x1 = np.sum(w1 * f(x1))
    
    # Compute the absolute errors
    errors_x0.append(abs(integral_x0 - exact_integral))
    errors_x1.append(abs(integral_x1 - exact_integral))

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogy(N_values, errors_x0, marker='o', label="Uniform Points")
plt.semilogy(N_values, errors_x1, marker='o', label="Chebyshev Points")
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Comparison of Quadrature Rules")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# **Discussion**:
# The quadrature rule using Chebyshev points converges much faster than the rule using uniformly spaced points. This is because Chebyshev points are designed to minimize interpolation error, which leads to more accurate numerical integration. For smooth functions like \(f(x)\), the Chebyshev-based rule is clearly superior, especially for larger values of \(N\).
