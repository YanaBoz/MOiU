import numpy as np
import re
from itertools import combinations

def generate_valid_bases(A):
    m, n = A.shape
    valid_bases = []
    
    for cols in combinations(range(n), m): 
        B = A[:, cols]  
        if np.linalg.det(B) != 0:
            valid_bases.append(cols)

    return valid_bases


def extract_coefficients(equations):
    num_equations = len(equations)
    variables = set()
    pattern = r"([+-]?\d*\.?\d+|[+-])?([a-zA-Z]\d*)"
    
    for equation in equations:
        terms = re.findall(pattern, equation)
        for coeff, var in terms:
            if var:
                variables.add(var)
    
    variables = sorted(list(variables))
    num_variables = len(variables)
    
    A = np.zeros((num_equations, num_variables))
    b = np.zeros(num_equations)
    
    for i, equation in enumerate(equations):
        terms = re.findall(pattern, equation)
        for coeff_str, var in terms:
            if var:
                coeff = -1.0 if coeff_str == "-" else (1.0 if coeff_str in ["", "+"] else float(coeff_str))
                var_index = variables.index(var)
                A[i, var_index] = coeff
        
        constant_term_match = re.search(r"=(.*)", equation)
        if constant_term_match:
            b[i] = float(constant_term_match.group(1))
        else:
            return None
    
    return A, b, variables


def linear_programming_canonical(equations, objective_function):
    A, b, variables = extract_coefficients(equations)
    
    obj_pattern = r"([+-]?\d*\.?\d*)*([a-zA-Z]\d*)"
    obj_terms = re.findall(obj_pattern, objective_function)
    
    c = np.zeros(len(variables))
    for coeff_str, var in obj_terms:
        if var:
            coeff = -1.0 if coeff_str == "-" else (1.0 if coeff_str in ["", "+"] else float(coeff_str))
            var_index = variables.index(var)
            c[var_index] = coeff
    
    basis = [i for i, col in enumerate(A.T) if sum(col) == 1 and set(col) <= {0, 1}]
    x_b = b.copy()
    x = np.zeros(len(variables))
    for i, var in enumerate(basis):
        if 0 <= var < len(x):  
           x[var] = x_b[i]
    
    return c, variables, A, b, x, basis


def modify_matrix(A, A_modified):
    n = A.shape[0]

    if A.shape != A_modified.shape:
        return "ERROR: Matrices A and A_modified must have the same dimensions."

    i = -1
    for col in range(n):
        if not np.allclose(A[:, col], A_modified[:, col]): 
            i = col + 1 
            break
    
    if i == -1:
        return "ERROR: No column differs between A and A_modified."
    
    x = A_modified[:, i-1] - A[:, i-1]
    x = x.reshape(-1, 1) 

    A1 = np.linalg.inv(A)
    
    l = np.dot(A1, x)
    
    if l[i-1, 0] != 0:
        y = -1
        l1 = l.copy()
        l1[i-1:] = y
        
        l2 = -1 / l[i-1] * l1
        
        Q = np.eye(n)
        
        Q_modified = Q.copy()
        Q_modified[:, i-1] = l2.flatten()
        
        A_modified1 = np.dot(Q_modified, A1)
        print("\nA_modified1:")
        print(A_modified1)
        return A_modified1
    else:
        return np.linalg.inv(A_modified)

objective_function = "x1 + x2"
equations = [
    "-x1 + x2 + x3 = 1",
    "x1 + x4 = 3",
    "x2 + x5 = 2"
]
count = 1

c, variables, A, b, x, basis = linear_programming_canonical(equations, objective_function)

valid_bases = generate_valid_bases(A)
print("Valid bases:", valid_bases)

if valid_bases:
    selected_basis_index = int(input(f"Select a basis (0 to {len(valid_bases) - 1}): "))
    basis = list(valid_bases[selected_basis_index])
    basis1 = [index + 1 for index in valid_bases[selected_basis_index]]
    print("Selected basis (with +1):", basis1)
else:
    print("No valid bases found.")

m, n = A.shape

print("Vector of objective function coefficients (c):", c)
print("Vector of variables:", variables)
print("Coefficient matrix A:")
print(A)
print("Vector b:", b)
print("Basic variables (indices):", basis1)
print("Initial basic feasible solution:", x)

while count <= 7:
    print(f"Iteration {count}")
    count += 1
    A_B = A[:, basis] 
    print("Matrix AB:")
    print(A_B)

    if count == 3:
        A_B_inv = modify_matrix(memory, A_B) 
        if isinstance(A_B_inv, str):  
            print(A_B_inv) 
            break 
    else:
        A_B_inv = np.linalg.inv(A_B) 
    memory = A_B
    A_B_inv = np.array(A_B_inv, dtype=np.float64)  

    print("Inverse matrix AB:")
    print(A_B_inv)

    c_B = c[basis]  

    c_B = np.array(c_B, dtype=np.float64)  

    print("Vector cB:")
    print(c_B)

    U_ = np.dot(c_B, A_B_inv)  

    print("Vector U:")
    print(U_)

    t = np.dot(U_, A) - c

    print("Vector t:")
    print(t)

    negative_components = [i for i, val in enumerate(t) if val < 0]

    if negative_components:
        j0 = negative_components[0] 
        print(f"First negative component index (j0): {j0}")
    else:
        print("No negative components in the reduced cost vector.")
        print("Optimal solution reached:", x)
        break

    del1 = t[j0]
    print(f"del1: {del1}")

    A_j0 = A[:, j0] 

    z = np.dot(A_B_inv, A_j0) 

    print("Vector z:")
    print(z)

    theta = np.full_like(z, np.inf)  

    for i in range(len(z)):
        if z[i] > 0:
            theta[i] = x[basis[i]] / z[i]  

    print("Vector θ:")
    print(theta)

    valid_thetas = [theta_i for theta_i in theta if theta_i != np.inf]

    if valid_thetas:
        theta_0 = min(valid_thetas)
        print(f"θ0 (minimum valid θ_i): {theta_0}")
    else:
        print("No valid θ_i found.")
        break 

    if theta_0 == np.inf:
        print("The objective function is unbounded on the feasible set.")
    else:
        print(f"θ0 (minimum valid θ_i): {theta_0}")

    min_theta_value = np.inf
    k = -1
    for i, theta_i in enumerate(theta):
        if theta_i < min_theta_value:  
            min_theta_value = theta_i
            k = i

    if k != -1:
        j_star = basis[k]  
        print(f"The first index k where the minimum occurs: {k}")
        print(f"Corresponding basis index j*: {j_star}")
    else:
        print("No valid θ_i found.")

    basis[k] = j0 

    print(f"Updated basis B: {basis}")

    x[j0] = theta_0 
    for i in range(m):
        if i != k:
            j_i = basis[i] 
            x[j_i] = x[j_i] - theta_0 * z[i]  

    x[j_star] = 0 

    print("Updated plan x:", x)