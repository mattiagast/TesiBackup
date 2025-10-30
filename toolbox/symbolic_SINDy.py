import numpy as np
import pysindy as ps
import dill
import os

from D_CODE.run_simulation import run as run_SRT
from D_CODE.run_simulation_vi import run as run_DCODE
from toolbox.auxiliary_functions import intercept_library_fun
from toolbox.auxiliary_functions import check_building_blocks, filter_building_blocks # Filter
from data.SINDy_data import evaluate_RMSE_d

# NOTE: Ulteriori possibili miglioramenti:
# - Scegliere il modello valutando piu threshold e.g [0.09, 0.20, 0.35]

# TODO: Possibili miglioramenti:
# - Gestire il caso no building blocks con il semplice SINDy 
# - Metodo 'update' che gestisce l'attributo model e bb metodo get_feature_names
# - Metodo 'print' per salvare il modello


class symbolic_SINDy(): 

    def __init__(self, SR_method='SR-T', x_id=0, alg='tv', degree=3, threshold=0.09, penalty=20, product=False):
        self.SR_method = SR_method # either 'SR-T' or 'D-CODE'
        self.x_id = x_id 
        self.alg = alg
        self.degree = degree
        self.threshold = threshold
        self.penalty = penalty
        self.product = product



    def SRT_simulation(self, ode_name, param, freq, n_sample, noise_sigma, seed, n_seed, T0, T):
        print('')
        print('Searching for additonal building blocks -> SR-T call:')
        print('')
        print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, alg={}, seed={}, n_seed={}".format(
        ode_name, param, self.x_id, freq, n_sample, noise_sigma, self.alg, seed, n_seed))
        building_blocks_lambda, function_names = run_SRT(ode_name, param, self.x_id, freq, n_sample, noise_sigma, self.alg, seed, n_seed, T0, T, idx=0)
        return building_blocks_lambda, function_names


    def D_CODE_simulation(self, ode_name, param, freq, n_sample, noise_sigma, seed, n_seed, T0, T):
        print('')
        print('Searching for additonal building blocks -> D-CODE call:')
        print('')
        print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, seed={}, n_seed={}".format(
        ode_name, param, self.x_id, freq, n_sample, noise_sigma, seed, n_seed))
        building_blocks_lambda, function_names = run_DCODE(ode_name, param, self.x_id, freq, n_sample, noise_sigma, seed, n_seed, T0, T, idx=0)
        return building_blocks_lambda, function_names



    def call(self, X_list, dX_list, param_list, feature_names, dt, building_blocks_lambda, function_names, patience, lazy, ode, ode_name, ode_param, freq_SR, n_sample, noise_ratio, seed, n_seed, T0, T, dim_x, dim_k):

        if building_blocks_lambda is None or patience > 4: # either SR has never been called or patience is over (the found building blocks have not been useful over last 5 iterations)

            if lazy == False: # -> compute the building blocks with Symbolic Regression
                if self.SR_method == 'SR-T':
                    # SR-T call:
                    building_blocks_lambda, function_names = self.SRT_simulation(ode_name, ode_param, freq_SR, n_sample, noise_ratio, seed, n_seed, T0, T)
                else:
                    # D-CODE call:
                    building_blocks_lambda, function_names = self.D_CODE_simulation(ode_name, ode_param, freq_SR, n_sample, noise_ratio, seed, n_seed, T0, T)
                # save building blocks: 
                file_path = f"saved/building_blocks/{ode_name}_bb.pkl"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    dill.dump((building_blocks_lambda, function_names), f)
            else: # -> load the building blocks of Symbolic Regression
                # load building blocks:
                file_path = f"saved/building_blocks/{ode_name}_bb.pkl"
                with open(file_path, 'rb') as f:
                    building_blocks_lambda, function_names = dill.load(f)
            patience = 0
            # building_blocks_lambda.append(lambda X0, X1, X2: np.sin(5*X2))
            # function_names.append(lambda X0, X1, X2: "sin(5*"+X2+")")

        # filter the building blocks
        building_blocks_lambda, function_names = check_building_blocks(X_list, building_blocks_lambda, function_names)
        building_blocks_lambda, function_names = filter_building_blocks(feature_names, building_blocks_lambda, function_names, self.degree)
        
        print('')
        print('Searching for the best building block:')
        patience += 1
        errors = []
        n_features_vec = []
        intercept_library = intercept_library_fun(dim_x+dim_k) # intercept library
        polynomial_library = ps.PolynomialLibrary(degree=self.degree, include_bias=False) # polynomial library

        if len(building_blocks_lambda) == 0:
            print("No useful building blocks found")
            return None, None, None, None, None, patience
        
        for i in range(len(building_blocks_lambda)):

            # building the library:
            custom_library = ps.CustomLibrary(library_functions=[building_blocks_lambda[i]], function_names=[function_names[i]]) # custom library with building block
            if self.product: # -> tensor product of the Polynomial and Custom 
                generalized_library = ps.GeneralizedLibrary(libraries=[polynomial_library, custom_library],tensor_array=[[1, 1]])
            else: # -> concatenation of the Polynomial and Custom
                generalized_library = ps.ConcatLibrary([polynomial_library, custom_library], ) # enlarged library, adding the building block to polynomial library
            final_library = ps.ConcatLibrary([intercept_library, generalized_library]) # add the intercept

            # fitting the model:
            model = ps.SINDy(feature_names=feature_names, feature_library=final_library, optimizer=ps.STLSQ(threshold=self.threshold))
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list)
            print('Model:')
            model.print()   

            # print('')
            # print('library:')
            # library_terms = final_library.get_feature_names(input_features=feature_names)
            # for term in library_terms:
            #     print(term)
            # print()   

            # evaluate the model:  
            coefficients = model.coefficients()
            model_complexity = np.count_nonzero(np.array(model.coefficients()))
            lasso_penalty = np.sum(np.abs(coefficients))
            if model_complexity < self.penalty and lasso_penalty < self.penalty: #filter too complex models (for sure not correct and likely to crash the code):
                _, mse = evaluate_RMSE_d(model, ode, 10, 10, ode.init_high, ode.init_low, T0, T, dim_k) # compute MSE      
                alpha = 0.01 # regularization parameter
                error = mse + alpha * lasso_penalty # final evaluation metric
                #print('error:', error)
            else:
                error = 1000
                #print('Too complex model')
            #print('')
            errors.append(error)
            n_features_vec.append(np.count_nonzero(np.array(model.coefficients())))

        print("errors: ", errors)
        if all(err == 1000 for err in errors):
            print('No model update, all smart-SINDy models are too complex')
            return None, building_blocks_lambda, function_names, None, None, patience
        else:
            # Final model
            min_error = min(errors)
            idxs = [i for i, e in enumerate(errors) if abs(e - min_error) < 0.01]
            n_features_vec_2 = [n_features_vec[i] for i in idxs]

            if len(idxs) > 1:
                # print('Multiple models with similar error, selecting the simplest one with the lowest error')
                # print('')
                min_features = min(n_features_vec_2) # find the min number of features among the candidates
                idxs_min_feat = [idxs[i] for i, nf in enumerate(n_features_vec_2) if nf == min_features] # select al the indexes with that number of features
                idx = idxs_min_feat[np.argmin([errors[i] for i in idxs_min_feat])] # among these, choose the one with less error
            else:
                idx = idxs[0]

            # building the library:
            custom_library = ps.CustomLibrary(library_functions=[building_blocks_lambda[idx]], function_names=[function_names[idx]])  # custom library with building block
            model = ps.SINDy(feature_names=feature_names, feature_library=custom_library, optimizer=ps.STLSQ(threshold=0.01))
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list)
            building_block = custom_library.get_feature_names(input_features=feature_names) 
            if self.product: # -> tensor product of the Polynomial and Custom 
                generalized_library = ps.GeneralizedLibrary(libraries=[polynomial_library, custom_library],tensor_array=[[1, 1]])
            else: # -> concatenation of the Polynomial and Custom
                generalized_library = ps.ConcatLibrary([polynomial_library, custom_library], ) # enlarged library, adding the building block to polynomial library
            final_library = ps.ConcatLibrary([intercept_library, generalized_library]) # add the intercept

            # fitting the model:
            model = ps.SINDy(feature_names=feature_names, feature_library=final_library, optimizer=ps.STLSQ(threshold=self.threshold))
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list)

            # best building block:
            print('')
            print('Best building block:')
            print(building_block)
            print('')

            # final model:
            print('smart-SINDy model:')
            model.print()

            # save model and bb: 
            file_path = f'saved/results/{ode_name}_model.pkl'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                dill.dump((model, building_block), f)

            coefficients = model.coefficients()
            model_complexity = np.count_nonzero(np.array(model.coefficients()))
            print('Model complexity: ', model_complexity)
            lasso_penalty = np.sum(np.abs(coefficients))
            print('Lasso penalty: ', lasso_penalty)

            return model, building_blocks_lambda, function_names, model_complexity, lasso_penalty, patience


    
            