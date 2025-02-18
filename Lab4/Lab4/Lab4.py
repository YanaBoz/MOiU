import numpy as np
import re
from fractions import Fraction

def transform_b(A, b):
    m = b.shape[0]
    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1
    return A, b

def extract_coefficients(equations):
    num_equations = len(equations)
    variables = set()
    pattern = r"([+-]?\d*\.?\d+|[+-])?([a-zA-Z]\d+)"
    
    for eq in equations:
        for _, var in re.findall(pattern, eq):
            if var:
                variables.add(var)
                
    variables = sorted(list(variables))
    A = np.zeros((num_equations, len(variables)))
    b = np.zeros(num_equations)
    
    for i, eq in enumerate(equations):
        for coeff_str, var in re.findall(pattern, eq):
            if var:
                coeff = -1.0 if coeff_str.strip() == "-" else (1.0 if coeff_str.strip() in ["", "+"] else float(coeff_str.strip()))
                A[i, variables.index(var)] = coeff
        
        match = re.search(r"=(.*)", eq)
        if match:
            rhs = match.group(1).strip()
            if rhs.startswith('-'):
                rhs = rhs[1:].strip()
            if '/' in rhs:
                b[i] = float(Fraction(rhs))
            else:
                b[i] = float(rhs)
        else:
            raise ValueError(f"Right-hand side not found in equation: {eq}")
    
    return A, b, variables

def parse_fraction(fraction_str):
    fraction_str = fraction_str.strip()
    if '/' in fraction_str:
        return float(Fraction(fraction_str))
    else:
        return float(fraction_str)

def parse_objective_function(obj_func, variables):
    pattern = r"([+-]?\d*\.?\d+|[+-])?([a-zA-Z]\d+)"
    coefficients = {var: 0.0 for var in variables}
    for coeff_str, var in re.findall(pattern, obj_func):
        coeff = -1.0 if coeff_str == "-" else (1.0 if coeff_str in ["", "+"] else float(coeff_str))
        if var in coefficients:
            coefficients[var] = coeff
    return coefficients

def add_artificial_variables(A, b):
    m, n = A.shape
    A_e = np.hstack((A, np.eye(m)))
    c_e = np.concatenate((np.zeros(n), -np.ones(m)))
    return A_e, b.copy(), c_e, n

def initial_basic_feasible_solution(b, n, m):
    x_e = np.zeros(n + m)
    x_e[n:] = b
    basis = list(range(n, n + m))
    return x_e, basis

def dual_simplex(A_e, c_e, b, n, m, tol=1e-8, max_iter=100):
    x_e, basis = initial_basic_feasible_solution(b, n, m)
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        B = A_e[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise ValueError("Базисная матрица не является обратимой.")
        c_B = c_e[basis]
        U = c_B.dot(B_inv)
        t = U.dot(A_e) - c_e

        if np.all(t >= -tol):
            print("Двойственная допустимость достигнута.")
            break
        
        j0 = np.where(t < -tol)[0][0]
        z = B_inv.dot(A_e[:, j0])
        
        if np.all(z <= tol):
            raise ValueError("Задача неограничена.")
        
        theta_vals = []
        for i in range(len(z)):
            theta_vals.append(x_e[basis[i]] / z[i] if z[i] > tol else np.inf)
        theta = min(theta_vals)
        leaving_index = theta_vals.index(theta)
        
        for i in range(len(basis)):
            x_e[basis[i]] = x_e[basis[i]] - theta * z[i]
        x_e[j0] = theta
        basis[leaving_index] = j0
    
    return x_e, basis

def check_compatibility(x_e, n, m):
    if np.all(np.abs(x_e[n:]) < 1e-8):
        print("Прямая задача допустима.")
        return True
    else:
        print("Прямая задача недопустима. Завершаем.")
        return False

def solve_lp_dual(equations, objective_function):
    A, b, variables = extract_coefficients(equations)
    A, b = transform_b(A, b)
    m, n_orig = A.shape

    obj_coeffs = parse_objective_function(objective_function, variables)

    c = np.zeros(n_orig)
    for var, coeff in obj_coeffs.items():
        if var in variables:
            c[variables.index(var)] = -coeff
        else:
            print(f"Переменная {var} из целевой функции не найдена в ограничениях.")

    print(f"Коэффициенты целевой функции: {c}")
    print(f"Матрица ограничений A: \n{A}")
    print(f"Правая часть b: {b}")
    
    print("Используем метод двойного симплекс.")
    A_e, b_aux, c_e, n = add_artificial_variables(A, b)
    x_e, basis = dual_simplex(A_e, c_e, b_aux, n, m)

    if not check_compatibility(x_e, n, m):
        return None

    x_formatted = tuple(float(val) for val in np.round(x_e, 4))

    optimal_solution = {variables[i]: x_e[i] for i in range(n_orig)}

    basis_str = f"j1 = {basis[0] + 1}" if basis else ""
    A_str = tuple(float(val) for val in np.round(A[0, :], 4))
    b_str = tuple(float(val) for val in np.round(b[:1], 4))
    answer = f"Ответ: x = {x_formatted}, Базис = {{{basis_str}}}, A = {A_str} и b = {b_str}."

    print(answer)
    print("Оптимальный план k (значения переменных):")
    for var, val in optimal_solution.items():
        print(f"{var} = {val:.4f}")
    
    return x_e, basis, A, b

if __name__ == '__main__':
    objective_function = "- 4x1 - 3x2 - 7x3 → max"
    equations = [
        "- 2x1 - x2 - 4x3 + x4 = -1",
        "- 2x1 - 2x2 - 2x3 + x5 = -3/2"
    ]
    solve_lp_dual(equations, objective_function)