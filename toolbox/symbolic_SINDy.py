import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import dill
import os
import time

from D_CODE.run_simulation import run as run_SRT
from D_CODE.run_simulation_vi import run as run_DCODE
from toolbox.auxiliary_functions import intercept_library_fun
from toolbox.auxiliary_functions import filter_complete, filter_complete_param # Filter
from data.SINDy_data import evaluate_RMSE_d

from toolbox.auxiliary_functions import sindy_from_coef # UQ models
from scipy.stats import gaussian_kde # UQ plot


# POSSIBILI MIGLIORAMENTI:
# 1) gestione del modello nullo: -> fare come quando il modello è troppo complesso
# 2) Nota che SR-T e D-CODE intrinsecamnete generano i dati al prioprio interno (nota: non lo fa run_HD)
#    Anche evaluate_RMSE_d genera la ground truth noise-free come reference per il calcolo del RMSE
#    (nota: vedi evaluate_RMSE_HD che valuta RMSE tra il modello trovato e i dati misurati)

class symbolic_SINDy(): 

    def __init__(self, SR_method='SR-T', x_id=0, alg='tv', degree=3, threshold=0.09, penalty=20, product=False, max_patience=4):
        self.SR_method = SR_method      # either 'SR-T' or 'D-CODE'
        self.x_id = x_id                # dimension on which SR is performed
        self.alg = alg                  # algebric differentiation method (for SR-T)
        self.degree = degree            # degree of the Polynomial CFL in SINDy
        self.threshold = threshold      # STLSQ threshold coefficient
        self.penalty = penalty          # maximum model complexity allowed
        self.product = product          # if True then tensor product in the CFL

        self.max_patience = max_patience # max patience before calling SR again in on-the-fly tests

        # Store the informations of the trained model:
        self.model = None               # Symbolic-SINDy model saved as SINDy object
        self.building_block = None      # optimal building block found
        self.cfl = None                 # CFL of the Symbolic-SINDy fit
        self.coefficients = None        # CFL of the Symbolic-SINDy model
        self.model_complexity = None    # number of the coefficient of the Symbolic-SINDy model
        self.lasso_penalty = None       # sum of all the coefficients

        self.building_blocks_lambda = None      # list of building block identified by SR as lambda functions
        self.function_names = None              # list of building block identified by SR as gplearn string

        # Store the informations for UQ
        self.coef_list = None           # list of inferred coefficients
        self.median_model = None        # median Symbolic-SINDy model as SINDy object
        self.lower_model = None         # 0.025 quantile Symbolic-SINDy model as SINDy object
        self.upper_model = None         # 0.975 quantile Symbolic-SINDy model as SINDy object


    def __update(self, model, building_block, final_library, coefficients, model_complexity, lasso_penalty):
        # Update method: it update the inner structure of the class
        self.model = model
        self.building_block = building_block
        self.cfl = final_library
        self.coefficients = coefficients
        self.model_complexity = model_complexity
        self.lasso_penalty = lasso_penalty
        return
    
    
    def print(self):
        # Print method: print the principal information of the model
        print('Additional block: ', self.building_block)
        print('Tensor product: ', self.product)
        # final model:
        print('Smart-SINDy model:')
        self.model.print()

        print('Model complexity: ', self.model_complexity)
        print('Lasso penalty: ', self.lasso_penalty)

        return


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

        if building_blocks_lambda is None or patience > self.max_patience: # either SR has never been called or patience is over (the found building blocks have not been useful over last 5 iterations)

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
        t0 = time.time()
        # filter the building blocks:
        building_blocks_lambda, function_names = filter_complete(X_list, feature_names, self.degree, building_blocks_lambda, function_names)

        print('')
        print('Searching for the best building block:')
        patience += 1
        errors = []
        n_features_vec = []
        intercept_library = intercept_library_fun(dim_x+dim_k) # intercept library
        polynomial_library = ps.PolynomialLibrary(degree=self.degree, include_bias=False) # polynomial library

        # if no useful building block -> SINDy standard
        if len(building_blocks_lambda) == 0:
            print("No valid building block found: -> Simple SINDy call:")
            model = ps.SINDy(feature_names=feature_names, feature_library=polynomial_library, optimizer=ps.STLSQ(threshold=self.threshold))
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list)
            print('SINDy Model:')
            model.print()   
            print('')
            # evaluate the model:  
            coefficients = model.coefficients()
            model_complexity = np.count_nonzero(np.array(model.coefficients()))
            lasso_penalty = np.sum(np.abs(coefficients))

            print('Model complexity: ', model_complexity)
            print('Lasso penalty: ', lasso_penalty)
            
            self.__update(model, None, polynomial_library, coefficients, model_complexity, lasso_penalty)

            return model, None, None, model_complexity, lasso_penalty, patience
        
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
            # print('Model:')
            # model.print()

            # print('')
            # print('library:')
            # library_terms = final_library.get_feature_names(input_features=feature_names)
            # for term in library_terms:
            #     print(term)
            # print()

            # model metrics:  
            coefficients = model.coefficients()
            model_complexity = np.count_nonzero(np.array(model.coefficients()))
            lasso_penalty = np.sum(np.abs(coefficients))

            # evaluating model metrics
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
            print('Smart-SINDy model:')
            model.print()

            # save model and bb: 
            file_path = f'saved/results/{ode_name}_model.pkl'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                dill.dump((model, building_block), f)

            # model metrics:
            coefficients = model.coefficients()
            model_complexity = np.count_nonzero(np.array(model.coefficients()))
            lasso_penalty = np.sum(np.abs(coefficients))

            print('Model complexity: ', model_complexity)
            print('Lasso penalty: ', lasso_penalty)

            # update the model internally
            self.__update(model, building_block, final_library, coefficients, model_complexity, lasso_penalty)
            t1 = time.time()
            print("Total time: ", t1-t0)
            return model, building_blocks_lambda, function_names, model_complexity, lasso_penalty, patience


    def call_param(self, X_list, dX_list, param_list, feature_names, dt, building_blocks_lambda, function_names, patience, lazy, ode, ode_name, ode_param, freq_SR, n_sample, noise_ratio, seed, n_seed, T0, T, dim_x, dim_k):

        if building_blocks_lambda is None or patience > self.max_patience: # either SR has never been called or patience is over (the found building blocks have not been useful over last 5 iterations)

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
        t0 = time.time()
        # filter the building blocks:
        building_blocks_lambda, function_names = filter_complete_param(X_list, param_list, feature_names, self.degree, dim_k, building_blocks_lambda, function_names)

        print('')
        print('Searching for the best building block:')
        patience += 1
        errors = []
        n_features_vec = []
        intercept_library = intercept_library_fun(dim_x+dim_k) # intercept library
        polynomial_library = ps.PolynomialLibrary(degree=self.degree, include_bias=False) # polynomial library

        if len(building_blocks_lambda) == 0:
            print("No valid building block found: -> Simple SINDy call:")
            model = ps.SINDy(feature_names=feature_names, feature_library=polynomial_library, optimizer=ps.STLSQ(threshold=self.threshold))
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list, u=param_list)
            print('SINDy Model:')
            model.print()   
            print('')
            # evaluate the model:  
            coefficients = model.coefficients()
            model_complexity = np.count_nonzero(np.array(model.coefficients()))
            lasso_penalty = np.sum(np.abs(coefficients))

            print('Model complexity: ', model_complexity)
            print('Lasso penalty: ', lasso_penalty)
            
            self.__update(model, None, polynomial_library, coefficients, model_complexity, lasso_penalty)

            return model, None, None, model_complexity, lasso_penalty, patience
        
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
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list, u=param_list)
            # print('Model:')
            # model.print()   

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
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list, u=param_list)
            building_block = custom_library.get_feature_names(input_features=feature_names) 
            if self.product: # -> tensor product of the Polynomial and Custom 
                generalized_library = ps.GeneralizedLibrary(libraries=[polynomial_library, custom_library],tensor_array=[[1, 1]])
            else: # -> concatenation of the Polynomial and Custom
                generalized_library = ps.ConcatLibrary([polynomial_library, custom_library], ) # enlarged library, adding the building block to polynomial library
            final_library = ps.ConcatLibrary([intercept_library, generalized_library]) # add the intercept

            # fitting the model (with u control params):
            model = ps.SINDy(feature_names=feature_names, feature_library=final_library, optimizer=ps.STLSQ(threshold=self.threshold))
            model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list, u=param_list)

            # best building block:
            print('')
            print('Best building block:')
            print(building_block)
            print('')

            # final model:
            print('Smart-SINDy model:')
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

            self.__update(model, building_block, final_library, coefficients, model_complexity, lasso_penalty)
            t1 = time.time()
            print("Total time: ", t1-t0)
            return model, building_blocks_lambda, function_names, model_complexity, lasso_penalty, patience


    def fit(self, X_list, dX_list, param_list, feature_names, dt, ode, ode_name, ode_param, freq_SR, n_sample, noise_ratio, seed, n_seed, T0, T, dim_x, dim_k):
        if param_list == []:
            print("Fitting Smart-SINDy model")
            final_model, self.building_blocks_lambda, self.function_names, _, _, _ = self.call(X_list=X_list, dX_list=dX_list, param_list=param_list, feature_names=feature_names, dt=dt,
                                                                                               building_blocks_lambda=None, function_names=None, patience=0, lazy=False,
                                                                                               ode=ode, ode_name=ode_name, ode_param=ode_param,
                                                                                               freq_SR=freq_SR, n_sample=n_sample, noise_ratio=noise_ratio,
                                                                                               seed=seed, n_seed=n_seed, T0=T0, T=T, dim_x=dim_x, dim_k=dim_k)
        else:
            print("Fitting Smart-SINDy model with control parameters")
            final_model, self.building_blocks_lambda, self.function_names, _, _, _ = self.call_param(X_list=X_list, dX_list=dX_list, param_list=param_list, feature_names=feature_names, dt=dt,
                                                                                                     building_blocks_lambda=None, function_names=None, patience=0, lazy=False,
                                                                                                     ode=ode, ode_name=ode_name, ode_param=ode_param,
                                                                                                     freq_SR=freq_SR, n_sample=n_sample, noise_ratio=noise_ratio,
                                                                                                     seed=seed, n_seed=n_seed, T0=T0, T=T, dim_x=dim_x, dim_k=dim_k) 

   
    def ensemble(self, X_list, dX_list, param_list, feature_names, dt, n_models = 500):

        if param_list == []:
            final_model = ps.SINDy(feature_names=feature_names, feature_library=self.cfl, optimizer=ps.STLSQ(threshold=self.threshold))
            final_model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list, ensemble=True, n_models = n_models)
        else:
            final_model = ps.SINDy(feature_names=feature_names, feature_library=self.cfl, optimizer=ps.STLSQ(threshold=self.threshold))
            final_model.fit(X_list, t=dt, multiple_trajectories=True, x_dot=dX_list, u=param_list, ensemble=True, n_models = n_models)

        coef_list = np.array(final_model.coef_list)   # (n_models, n_features, n_states)
        self.coef_list = coef_list

        median = np.median(coef_list, axis=0)
        lower = np.percentile(coef_list, 2.5, axis=0)   # (n_states, n_features)
        upper = np.percentile(coef_list, 97.5, axis=0) 

        self.median_model = sindy_from_coef(median.T, self.cfl, feature_names)
        self.lower_model = sindy_from_coef(lower.T, self.cfl, feature_names)
        self.upper_model = sindy_from_coef(upper.T, self.cfl, feature_names)


    def uq_plot(self, feature_names=None, eps_var=1e-14):
        
        eps_var = 1e-14
        coef_list = np.array(self.coef_list)  # (n_models, n_states, n_features)
        n_models, n_states, n_features = coef_list.shape
        lib = self.model.get_feature_names()
        coef_median = np.median(coef_list, axis=0)  # (n_states, n_features)

        # Frequenza di inclusione dei termini
        inclusion = np.mean(coef_list != 0, axis=0)  # (n_states, n_features)

        # Filtro: se tutti gli stati di una feature sono nulli -> non plottarla
        var_per_feature = np.var(coef_list, axis=0)  # (n_states, n_features)
        valid_features_mask = np.any(var_per_feature > eps_var, axis=0)
        valid_features_idx = np.where(valid_features_mask)[0]

        lib_valid = [lib[i] for i in valid_features_idx]
        coef_median_valid = coef_median[:, valid_features_idx]
        n_features_valid = len(valid_features_idx)

        # Plot
        fig, axes = plt.subplots(
            n_features_valid,
            n_states,
            figsize=(4 * n_states, 0.6 * n_features_valid),
            squeeze=False
        )

        # Range comune
        global_ran = np.max(np.abs(coef_list[:, :, valid_features_idx])) + 0.2
        x_common = np.linspace(-global_ran, global_ran, 300)

        for idx_plot, f in enumerate(valid_features_idx):
            for s in range(n_states):
                ax = axes[idx_plot, s]
                coeff = coef_list[:, s, f]
                inc = inclusion[s, f]
                alpha_val = 0.3 + 0.7 * inc

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])

                var = np.var(coeff)

                if var > eps_var:
                    try:
                        std = np.std(coeff)
                        # print(std)
                        if std > 0:
                            kde = gaussian_kde(coeff)
                            y_vals = kde(x_common)

                            ax.plot(x_common, y_vals, linewidth=1, alpha=alpha_val)
                            ax.fill_between(x_common, y_vals, alpha=0.3 * alpha_val)
                        else:
                            # Caso completamente degenere (fallback)
                            c = float(np.mean(coeff))
                            ax.axvline(c, 0, 1, linewidth=1, alpha=alpha_val)

                    except Exception:
                        pass

                else:
                    c = float(np.mean(coeff))
                    ax.axvline(c, 0, 1, linewidth=1, alpha=alpha_val)
                    ax.set_xlim(-global_ran, global_ran)
                    ax.set_ylim(-0.05, 1.2)

                # Mediana (triangolino)
                median_val = coef_median[s, f]
                ax.plot(
                    median_val, 0.1,
                    marker='v',
                    markersize=3,
                    transform=ax.get_xaxis_transform()
                )

                # Tick solo ultima riga
                if idx_plot < n_features_valid - 1:
                    ax.set_xticklabels([])
                else:
                    ax.tick_params(axis='x', labelsize=5)

        # Titoli colonne (stati)
        if feature_names is None:
            feature_names = [f"State {i}" for i in range(n_states)]

        for s in range(n_states):
            axes[0, s].set_title(feature_names[s], fontsize=4, pad=20)

        # === Etichette della libreria (fig.text) ===
        for idx_plot, f in enumerate(valid_features_idx):
            pos = axes[idx_plot, 0].get_position()
            y_center = pos.y0 + pos.height / 2

            fig.text(
                0.02, y_center,
                lib[f],
                va='center',
                ha='right',
                fontsize=4
            )

        # Spaziatura finale
        plt.subplots_adjust(left=0.15, hspace=0.4, wspace=0.2)
        # plt.savefig("hill_wrong.png", dpi=300, bbox_inches="tight")
        plt.show()


        

