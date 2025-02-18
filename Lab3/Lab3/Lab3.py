import numpy as np
import re
from itertools import combinations

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
    pattern = r"([+-]?\d*\.?\d+|[+-])?([a-zA-Z]\d*)"
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
                coeff = -1.0 if coeff_str == "-" else (1.0 if coeff_str in ["", "+"] else float(coeff_str))
                A[i, variables.index(var)] = coeff
        match = re.search(r"=(.*)", eq)
        if match:
            b[i] = float(match.group(1))
        else:
            raise ValueError(f"Не найдена правая часть в уравнении: {eq}")
    return A, b, variables

def add_artificial_variables(A, b):
    m, n = A.shape
    A_e = np.hstack((A, np.eye(m)))
    c_e = np.concatenate((np.zeros(n), -np.ones(m)))
    return A_e, b.copy(), c_e, n

def generate_valid_bases(A):
    m, n = A.shape
    valid_bases = []
    for cols in combinations(range(n), m):
        B = A[:, cols]
        if np.linalg.det(B) != 0:
            valid_bases.append(cols)
    return valid_bases

def initial_basic_feasible_solution(b, n, m):
    x_e = np.zeros(n + m)
    x_e[n:] = b
    basis = list(range(n, n + m))
    return x_e, basis

def simplex_phase1(A_e, c_e, b, n, m, tol=1e-8, max_iter=100):
    x_e, basis = initial_basic_feasible_solution(b, n, m)
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        B = A_e[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            raise ValueError("Базисная матрица необратима")
        c_B = c_e[basis]
        U = c_B.dot(B_inv)
        t = U.dot(A_e) - c_e
        if np.all(t >= -tol):
            break
        j0 = np.where(t < -tol)[0][0]
        z = B_inv.dot(A_e[:, j0])
        if np.all(z <= tol):
            raise ValueError("Задача неограничена (фаза 1)")
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
        print("Задача (1) совместна.")
        return True
    else:
        print("Задача (1) не совместна. Метод завершает работу.")
        return False

def form_feasible_plan(x_e, basis, n, m):
    print("Формируем допустимый план x для исходной задачи.")
    return x_e[:n], basis

def adjust_basis(A_e, basis, n, m):
    B = A_e[:, basis]
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise ValueError("Базисная матрица необратима при корректировке базиса.")
    for i in range(m):
        jk = n + i
        if jk in basis:
            for j in range(n):
                if j not in basis:
                    ell_j = B_inv.dot(A_e[:, j])
                    if abs(ell_j[i]) > 1e-8:
                        basis[basis.index(jk)] = j
                        B = A_e[:, basis]
                        B_inv = np.linalg.inv(B)
                        break
    return basis

def normalize_constraints(A, b):
    m = A.shape[0]
    r = np.linalg.matrix_rank(A)
    new_A = A[:r, :]
    new_b = b[:r].copy()
    for i in range(r):
        if abs(new_b[i]) < 1e-8:
            new_b[i] = 1.0
    return new_A, new_b

def solve_lp(equations, objective_function):
    A, b, variables = extract_coefficients(equations)
    A, b = transform_b(A, b)
    m, n_orig = A.shape

    print("Применяется двухфазный метод (искусственные переменные).")
    A_e, b_aux, c_e, n = add_artificial_variables(A, b)
    x_e, basis = simplex_phase1(A_e, c_e, b_aux, n, m)
    if not check_compatibility(x_e, n, m):
        return None
    x, chosen_basis = form_feasible_plan(x_e, basis, n, m)
    chosen_basis = adjust_basis(A_e, chosen_basis, n, m)
    
    r = np.linalg.matrix_rank(A)
    if m > r:
        A, b = normalize_constraints(A, b)
    
    x_formatted = tuple(float(val) for val in np.round(x, 4))
    if chosen_basis:
        basis_str = f"j1 = {chosen_basis[0] + 1}"
    else:
        basis_str = ""
    A_str = tuple(float(val) for val in np.round(A[0, :], 4))
    b_str = tuple(float(val) for val in np.round(b[:1], 4))
    answer = f"Ответ: x = {x_formatted}, B = {{{basis_str}}}, A = {A_str} и b = {b_str}."
    print(answer)
    return x, chosen_basis, A, b

if __name__ == '__main__':
    objective_function = "x1 → max"
    equations = [
        "x1 + x2 + x3 = 0",
        "2x1 + 2x2 + 2x3 = 0"
    ]
    solve_lp(equations, objective_function)