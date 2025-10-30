# Libraries:
import numpy as np
import sympy as sp
import pysindy as ps
import dill
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from D_CODE.run_simulation import run as run_SRT
from D_CODE.run_simulation_vi import run as run_DCODE
# from D_CODE.run_simulation_vi import run_HD as run_DCODE_HD
from data import SINDy_data


def set_param_freq(ode_param, freq):
    if ode_param is not None:
        param = [float(x) for x in ode_param.split(',')]
    else:
        param = None

    if freq >= 1:
        freq = int(freq)
    else:
        freq = freq

    return param, freq


def SRT_simulation(ode_name, param, x_id, freq, n_sample, noise_sigma, alg, seed, n_seed, idx=0, T0=0, T=15):
    print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, alg={}, seed={}, n_seed={}".format(
    ode_name, param, x_id, freq, n_sample, noise_sigma, alg, seed, n_seed))
    building_blocks_lambda, function_names = run_SRT(ode_name, param, x_id, freq, n_sample, noise_sigma, alg, seed=seed, n_seed=n_seed, idx=idx, T0=T0, T=T)
    return building_blocks_lambda, function_names


def D_CODE_simulation(ode_name, param, x_id, freq, n_sample, noise_sigma, seed, n_seed, T0=0, T=15):
    print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, seed={}, n_seed={}".format(
    ode_name, param, x_id, freq, n_sample, noise_sigma, seed, n_seed))
    building_blocks_lambda, function_names = run_DCODE(ode_name, param, x_id, freq, n_sample, noise_sigma, seed=seed, n_seed=n_seed, T0=T0, T=T)
    return building_blocks_lambda, function_names


def intercept_library_fun(n_variables):
    
    if n_variables == 1:
        X0 = sp.symbols('X0')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0), intercept, modules='numpy')
        function_name_intercept = lambda X0: str(intercept)    
    elif n_variables == 2:
        X0, X1 = sp.symbols('X0 X1')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1: str(intercept)
    elif n_variables == 3:
        X0, X1, X2 = sp.symbols('X0 X1 X2')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1, X2), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1, X2: str(intercept)        
    elif n_variables == 4:
        X0, X1, X2, X3 = sp.symbols('X0 X1 X2 X3')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1, X2, X3), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1, X2, X3: str(intercept)

    intercept_library = ps.CustomLibrary(library_functions=[intercept_lambda], function_names=[function_name_intercept])
    return intercept_library



def bb_combinations(building_blocks_lambda_0, building_blocks_lambda_1, function_names_0, function_names_1, init_high, init_low, dim_x, dim_k):

    # remove repeated building blocks:
    N = 10
    tol = 1e-3

    if dim_x == 1:
        x_samples = np.random.uniform(init_low, init_high, N).reshape(-1, 1) 
    else:
        x_samples = np.empty((dim_x+dim_k, N))
        for i in range(dim_x+dim_k):
            x_samples[i, :] = np.random.uniform(init_low[i], init_high[i], N) 
    x_samples = np.array(x_samples)
    #print(np.shape(x_samples))

    f_samples = []
    for i in range(len(building_blocks_lambda_0)):
        f_hat = building_blocks_lambda_0[i]
        #aux = [f_hat(x_samples[0, i], x_samples[1, i], x_samples[2, i]) for i in range(x_samples.shape[1])]
        aux = [f_hat(*x_samples[:, i]) for i in range(x_samples.shape[1])]
        f_samples.append(aux)
    #print(np.shape(f_samples))

    building_blocks_lambda_1_fil = [] # filter building blocks from eq. 2
    function_names_1_fil = []
    for i in range(len(building_blocks_lambda_1)):

        flag = 1
        f_hat = building_blocks_lambda_1[i] 
        #aux = [f_hat(x_samples[0, i], x_samples[1, i], x_samples[2, i]) for i in range(x_samples.shape[1])]
        aux = [f_hat(*x_samples[:, i]) for i in range(x_samples.shape[1])]
        for j in range(len(f_samples)):
            if root_mean_squared_error(aux, f_samples[j]) < tol: # filter similar subprograms
                flag = 0
        if flag:
            f_samples.append(aux)
            building_blocks_lambda_1_fil.append(building_blocks_lambda_1[i])
            function_names_1_fil.append(function_names_1[i])


    # combine building blocks from eq. 1 and eq. 2:
    combined_bb = [
    [lambda_0, lambda_1] 
    for lambda_0 in building_blocks_lambda_0 
    for lambda_1 in building_blocks_lambda_1_fil
    ]
    combined_fn = [
        [fn_0, fn_1] 
        for fn_0 in function_names_0 
        for fn_1 in function_names_1_fil
    ]
    bb_0 = [
        [lambda_0] 
        for lambda_0 in building_blocks_lambda_0 
    ]
    fn_0 = [
        [fn_0] 
        for fn_0 in function_names_0 
    ]
    bb_1 = [
        [lambda_1] 
        for lambda_1 in building_blocks_lambda_1_fil
    ]
    fn_1 = [
        [fn_1] 
        for fn_1 in function_names_1_fil
    ]

    bbs = bb_0 + bb_1 + combined_bb
    fns = fn_0 + fn_1 + combined_fn

    return bbs, fns


# Filter of the building blocks:
# Mettili in auxiliary functions
# Filter 
from sympy import symbols, simplify 
from itertools import combinations_with_replacement
import sympy as sp

def check_building_blocks(X_list, building_blocks_lambda, function_names):
    X_all = np.concatenate(X_list, axis=0)   # shape (n_traj*n_time, n_dim)
    valid_functions = [] 
    valid_names = []

    for i in range(len(building_blocks_lambda)):
        # Evaluate each building block in X_list
        vals = building_blocks_lambda[i](*[X_all[:, j] for j in range(X_all.shape[1])])
        # Check weather it produces a NaN or an invalid output
        if np.all(np.isfinite(vals)):  
            valid_functions.append(building_blocks_lambda[i])
            valid_names.append(function_names[i])   

    return valid_functions, valid_names 

def generate_monomials(feature_names, degree):
    """
    Genera tutti i monomi fino a 'degree' dati i nomi delle feature.
    Restituisce una lista di espressioni SymPy.
    """

    vars_sym = symbols(feature_names)
    monomials = []

    for d in range(1, degree + 1):
        # combina con ripetizione le variabili per generare tutti i monomi di grado d
        for combo in combinations_with_replacement(vars_sym, d):
            term = sp.Mul(*combo)
            monomials.append(term)
    return monomials

def is_monomial_to_remove(f, monomials, feature_names):
    """
    Controlla se la lambda f è un monomio nella lista monomials
    (anche moltiplicato per un coefficiente numerico)
    """
    vars_sym = symbols(feature_names)
    try:
        expr = simplify(f(*vars_sym))
        # separa coefficiente numerico e termine
        coeff, term = expr.as_coeff_Mul()
        return term in monomials
    except:
        return False

def filter_building_blocks(feature_names, building_blocks_lambda, function_names, degree):
    """
    Filtra i lambda in building_blocks_lambda rimuovendo i monomi fino a 'degree'.
    Se fornita, ritorna anche i function_names corrispondenti filtrati.
    """
    monomials_to_remove = generate_monomials(feature_names, degree)

    filtered_blocks = []
    filtered_names = []
    for f, name in zip(building_blocks_lambda, function_names):
        if not is_monomial_to_remove(f, monomials_to_remove, feature_names):
            filtered_blocks.append(f)
            filtered_names.append(name)
    return filtered_blocks, filtered_names