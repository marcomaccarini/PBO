import numpy as np
import PreferenceOptimization3.utils.math
import os
from datetime import datetime
import pandas as pd
import copy
from PreferenceOptimization3.model.RBF_modelF import RBFModelFinale
from PreferenceOptimization3.evaluators.batch_sequential import BatchSequential
from PreferenceOptimization3.variable.continuous_variable import ContinuousVariable
from PreferenceOptimization3.variable.discrete_variable import DiscreteVariable
from PreferenceOptimization3.variable.categorical_variable import CategoricalVariable
import PreferenceOptimization3.utils.math
from PreferenceOptimization3.utils.math import one_hot_to_cat, denormalize_X, normalize_X, scaleValues
import inspect
import glob
from PreferenceOptimization3.generator.anchor_points_generator.random_anchor_points import RandomAnchorPointsGenerator
from PreferenceOptimization3.optimizer.LBFGSB import LBFGSB
from PreferenceOptimization3.optimizer.SLSQP import SLSQP
from PreferenceOptimization3.optimizer.DE import DE
from PreferenceOptimization3.optimizer.Powell import Powell
from PreferenceOptimization3.optimizer.NelderMead import NelderMead
from PreferenceOptimization3.acquisition.Preferences import Preferences
from numpy import linalg as LA

class GLISpFinale(object):
    def __init__(self, fvars_x, fvars_y, express_preference=None, express_class_preference=None, objectives=None,
                 objectivesFunctions=None,
                 g=None, delta=1, acquisition_type='Preferences',
                 acquisition_optimizer_type='lbfgsb',
                 init_n_samples=5, samples_generator='random',
                 batch_size=1, kfold=None, save_experiment=False,
                 load_experiment_foldername=None,
                 load_experiment_foldername_class=None, my_fun_class=None,
                 name_opt='default',
                 model_type="RBF", evaluator_type='sequential_batch',
                 normalize_X=True, models=None, plotAcquisition=False,X=None, Y_b_in=None, Y_b_eq=None, Y_ind_best=None,
                 **kwargs):
        # region fill del costruttore

        if my_fun_class is None:
            my_fun_class = []
        self.init = 1  # todo: in teoria init dovrebbe settare l'esplorazione (?)
        self.g = g
        # self.maxBound = max_bound
        self.X_batch = None

        self.nameOpt = name_opt
        self.plotAcquisition = plotAcquisition

        self.my_fun_class = my_fun_class
        self.express_class_preference = express_class_preference

        self.delta = delta
        self.express_preference = express_preference
        self.acquisition_type = acquisition_type
        self.acquisition_optimizer_type = acquisition_optimizer_type
        # self.g = g
        self.init_n_samples = init_n_samples
        self.samples_generator_type = samples_generator
        # self.normalize_Y = normalize_Y
        self.batch_size = batch_size
        self.kfold = kfold
        self.kwargs = kwargs

        self.load_experiment_foldername = load_experiment_foldername
        self.save_experiment = save_experiment

        self.model_type = model_type

        self.ratio = self._calculate_ratio(fvars_x, fvars_y)
        # todo: questo verrà messo meglio
        # self.a0 = self._init_domain([fvars_x[0]])
        # self.a1 = self._init_domain([fvars_x[1]])
        # self.fvars_cat0, self.fvars_not_cat0 = self._split_variables(self.a0)
        # self.fvars_cat1, self.fvars_not_cat1 = self._split_variables(self.a1)
        # self.a0 = self.fvars_not_cat0 + self.fvars_cat0
        # self.a1 = self.fvars_not_cat1 + self.fvars_cat1

        self.fvars_x_Ratio = self._init_domain_Ratio(fvars_x, fvars_y, self.ratio)
        self.fvars_cat_Ratio, self.fvars_not_cat_Ratio = self._split_variables_Ratio(self.fvars_x_Ratio)

        self.fvars_x = self._init_domain(fvars_x)
        self.fvars_cat, self.fvars_not_cat = self._split_variables(self.fvars_x)
        self.fvars_x = self.fvars_not_cat + self.fvars_cat

        self.load_experiment_foldername_class = load_experiment_foldername_class
        self.folder_name = None
        self.n_class = len(my_fun_class)

        self.evaluator_type = evaluator_type

        self.fvars_y = fvars_y
        self.models = models


        self.X = None
        self.Y_b_in = None
        self.Y_b_eq = None
        self.Y_ind_best = None
        self.X_next = None
        self.X_next_all = None
        self.n_iteration = 0

        self.objectives = objectives
        self.objectivesFunctions = objectivesFunctions

        self.anchor_points_samples = 1000

        # todo: gestire la normalizzazione
        self.normalize_X = normalize_X

        nvars = len(self.fvars_not_cat)  # numero di variabili non categoriche
        self.bnd = np.zeros((nvars, 2))  # istanziazione dei bounds
        self.bnd_all = copy.copy(self.bnd)  # Dario dice: should be fixed!! ??

        self.acquisition = self.__acquisition_chooser()

        self.acquisition_optimizer = self.__acquisition_optimizer_chooser()

        self.anchor_points_generator = self.__anchor_points_chooser()

        self.evaluator = self._evaluator_chooser()

        for ind in range(nvars):
            aa = self.fvars_not_cat[ind]
            self.bnd[ind, :] = aa.bounds

        if self.normalize_X:
            for ind in range(nvars):
                aa = self.fvars_not_cat[ind]
                self.bnd_all[ind, :] = np.array([0, 1])

        # adding model to model's array
        self.models = self.modelsCreation(models, self.fvars_y)

        # load or generate data, their b_in and b_eq matrices and x_best array

        if X is not None and Y_b_in is not None and Y_b_eq is not None and Y_ind_best is not None:
            self.X = X
            self.Y_b_in = Y_b_in
            self.Y_b_eq = Y_b_eq
            self.Y_ind_best = Y_ind_best
        else:
            if self.load_experiment_foldername is not None:
                self.X, self.Y_b_in, self.Y_b_eq, self.Y_ind_best = self._load_experiment()
            else:  # Generate random samples
                self.X = self._generate_samples()
                self.Y_b_in, self.Y_b_eq, self.Y_ind_best = self._calculate_Y()

        # normalize data
        if self.normalize_X:
            self.X_all = PreferenceOptimization3.utils.math.normalize_X(self.fvars_x, self.X)
        else:
            self.X_all = self.X

        if load_experiment_foldername_class is not None:
            self._load_experiment_class()

        print(self.delta)

    """ Variables setup functions """

    def _split_variables_Ratio(self, fvars_x_Ratio):
        """
        Used to separate the variables.
        :param fvars: list of x
        :return: categorical variable's array and non_categorical variable's array
        """
        fvars_cat_Ratio = []
        fvars_not_cat_Ratio = []
        for fvars_x in fvars_x_Ratio:
            fvars_cat = []
            fvars_not_cat = []
            for fvar in fvars_x:
                if fvar.get_type() == 'categorical':
                    fvars_cat.append(fvar)
                else:
                    fvars_not_cat.append(fvar)
            fvars_cat_Ratio.append(fvars_cat)
            fvars_not_cat_Ratio.append(fvars_not_cat)
        return fvars_cat_Ratio, fvars_not_cat_Ratio

    def _calculate_ratio(self, fvars_x, fvars_y):
        if len(fvars_x) % len(fvars_y) != 0:
            raise NotImplementedError(f"Models not supported")
        else:
            return int(len(fvars_x) / len(fvars_y))

    def _init_domain_Ratio(self, fvars_x, fvars_y, ratio):
        array = []
        for i in range(len(fvars_y)):
            appoggio = fvars_x[i * ratio:(i + 1) * ratio]
            variables_list = []
            for function_var in appoggio:
                if function_var['type'] == 'continuous':
                    variables_list.append(ContinuousVariable(function_var['name'], np.array(function_var['domain'])))
                elif function_var['type'] == 'discrete':
                    variables_list.append(DiscreteVariable(function_var['name'], np.array(function_var['domain'])))
                elif function_var['type'] == 'categorical':
                    variables_list.append(CategoricalVariable(function_var['name'], np.array(function_var['domain'])))
                else:
                    raise Exception('Variable type not recognized:' + function_var['type'])
            array.append(variables_list)
        return array

    def _load_experiment(self):
        """
        Given a specified name of a folder, this procedure takes data from it. The folder's structure is:
        Experiment
            info.txt
            Data_X.csv
            best_X.csv
            Y
                Model0Data_Y.csv
                Model1Data_Y.csv
                ModelNData_Y.csv
        info.txt: contains hidden function, for each y in f_vars_y there will be a hiddend function and its model also.
        Data_X.csv: contains tested points. It's a simple array, each row contains a point that was tested.
        best_X.csv: contains the index's best point for each model.
        Y: contains ∏(x1,x2)
        :return: X, b_in, b_eq, ind_best.
        """
        # parsing X
        s_X = self.load_experiment_foldername + "/Data_X.csv"
        df_X = pd.read_csv(s_X, header=None)

        # parsing y
        all_files = glob.glob(self.load_experiment_foldername + "/Y/*.csv")
        all_files = sorted(all_files)

        # check for multiple y
        if len(self.fvars_y) != len(all_files):
            raise ValueError("f_varsY and data in Y's file not correspond")

        # parsing y
        appoggio_b_in = []
        appoggio_b_eq = []
        for filename in all_files:
            s_Y = filename
            df_Y = pd.read_csv(s_Y, header=None, names=["Xbest", "Xworst", "b"])
            mask_in = df_Y['b'] == 1
            mask_eq = df_Y['b'] == 0
            appoggio_b_in.append(df_Y[mask_in][["Xbest", "Xworst"]].values.astype(int))
            appoggio_b_eq.append(df_Y[mask_eq][["Xbest", "Xworst"]].values.astype(int))

        # parsing best
        s_ind = self.load_experiment_foldername + "/best_X.csv"
        df_ind = pd.read_csv(s_ind, header=None)
        return df_X.values, appoggio_b_in, appoggio_b_eq, df_ind.values.astype(int).ravel()

    def modelsCreation(self, models, fvars_y):
        """
        Creates for each Y a model
        :param models: for future implementation
        :param fvars_y: list of y
        :return: an array of models
        """
        arrModels = []
        if models is None:
            for i in range(len(fvars_y)):
                arrModels.append(self._model_chooser(i))
        else:
            for model in models:
                # arrModels.append(Model.from_dict(model))
                a = 0
                # todo: load model implementation
        return arrModels

    def _init_domain(self, domain_x):
        """
        Creation of variable instances from fvarsX
        :param domain_x: list of x
        :return:
        """
        variables_list = []
        for function_var in domain_x:
            if function_var['type'] == 'continuous':
                variables_list.append(ContinuousVariable(function_var['name'], np.array(function_var['domain'])))
            elif function_var['type'] == 'discrete':
                variables_list.append(DiscreteVariable(function_var['name'], np.array(function_var['domain'])))
            elif function_var['type'] == 'categorical':
                variables_list.append(CategoricalVariable(function_var['name'], np.array(function_var['domain'])))
            else:
                raise Exception('Variable type not recognized:' + function_var['type'])
        return variables_list

    def _split_variables(self, fvars):
        """
        Used to separate the variables.
        :param fvars: list of x
        :return: categorical variable's array and non_categorical variable's array
        """
        fvars_cat = []
        fvars_not_cat = []
        for fvar in fvars:
            if fvar.get_type() == 'categorical':
                fvars_cat.append(fvar)
            else:
                fvars_not_cat.append(fvar)
        return fvars_cat, fvars_not_cat

    def _generate_samples(self):
        """
        It creates point randomly
        :return: the list of points
        """
        # todo: mask
        x_init = np.empty((self.init_n_samples, 0))
        for function_var in self.fvars_x:
            if function_var.get_type() == 'continuous':
                x = np.random.uniform(function_var.get_domain()[0], function_var.get_domain()[1],
                                      (self.init_n_samples, 1))
            elif function_var.get_type() == 'discrete':
                x = np.random.choice(function_var.get_domain(), (self.init_n_samples, 1))
            elif function_var.get_type() == 'categorical':
                x = function_var.get_one_hot(np.random.choice(function_var.get_domain(), (self.init_n_samples, 1)))
            else:
                raise ValueError('Variable type not supported')
            x_init = np.column_stack((x_init, x))
        return x_init

    def _calculate_Y(self):
        """
        Used in the primary phase. At the beginning of any trial experiment (the hidden function is known) it has to create some initial points (>2).
        This method calculates comparison matrices by comparing initial points and by calculating the best.
        :return:
        """
        b_in_global = []
        b_eq_global = []
        ind_best_global = []

        for i in range(len(self.models)):
            N = np.shape(self.X[:, i])[0]
            b_in = np.zeros((N - 1, 2))
            ind_best = 0
            appoggio = self.objectives[i](self.X[:, i * self.ratio:(i + 1) * self.ratio])
            for ind in range(1, N):
                ind2 = ind - 1
                one = appoggio[ind2:ind2 + 1]
                two = appoggio[ind:ind + 1]
                best = appoggio[ind_best:ind_best + 1]
                if one <= two:
                    b_in[ind - 1, :] = [ind2, ind]
                    if one < best:
                        ind_best = ind2
                else:
                    b_in[ind - 1, :] = [ind, ind2]
                    if two < best:
                        ind_best = ind
            b_in_global.append(b_in)
            b_eq_global.append(np.array([]))
            ind_best_global.append(ind_best)
        return b_in_global, b_eq_global, ind_best_global

    def _load_experiment_class(self):
        # todo: sistemo load class experiment
        self.GP_class = {'X': [], 'Y': []}
        for i in range(self.n_class):
            s_X = self.load_experiment_foldername_class + "/Data_X_class" + str(i) + ".csv"
            s_Y = self.load_experiment_foldername_class + "/Data_Y_class" + str(i) + ".csv"
            df_X = pd.read_csv(s_X, header=None)
            df_Y = pd.read_csv(s_Y, header=None)
            self.GP_class['X'].append(df_X.values)
            self.GP_class['Y'].append(np.reshape(df_Y.values, (-1,)))

    def _save_experiment(self):
        """
        Used to save the experiment. It saves data in a given path (that it's stored in self.folder_name).
        The folder's structure is:
        Experiment
            info.txt
            Data_X.csv
            best_X.csv
            Y
                Model0Data_Y.csv
                Model1Data_Y.csv
                ModelNData_Y.csv
        info.txt: contains hidden function, for each y in f_vars_y there will be a hiddend function and its model also.
        Data_X.csv: contains tested points. It's a simple array, each row contains a point that was tested.
        best_X.csv: contains the index's best point for each model.
        Y: contains ∏(x1,x2)
        :return:
        """
        if self.save_experiment:
            folder_exists = os.path.exists(self.folder_name)
            # check if foldername exists
            if not folder_exists:
                os.makedirs(self.folder_name)
                os.makedirs(self.folder_name + "/Y")
                os.makedirs(self.folder_name + "/info")
                lines = ''
                con = 0
                if self.objectivesFunctions[0] is not None:
                    for e in self.objectivesFunctions:
                        lines += "FUNCTION EVAL #" + str(con) + ": \n"
                        con += 1
                        lines += inspect.getsource(e) + "\n"
                    s = self.folder_name + '/info/info.txt'
                    text_file = open(s, "w")
                    text_file.write(lines)
                    text_file.close()
            else:
                os.remove(self.folder_name + '/Data_X.csv')
                fileList = glob.glob(self.folder_name + '/Y/*.csv')
                for filePath in fileList:
                    os.remove(filePath)
                # os.remove(self.folder_name + '/Y/*.csv')
                os.remove(self.folder_name + '/best_X.csv')

            # saving X
            s = self.folder_name + '/Data_X.csv'
            np.savetxt(s, self.X, delimiter=',')

            # saving bests
            s = self.folder_name + '/best_X.csv'
            np.savetxt(s, self.Y_ind_best, delimiter=',')

            # saving comparisons
            for contatore in range(len(self.models)):
                M = np.shape(self.Y_b_in[contatore])[0]
                s = self.folder_name + '/Y/Model' + str(contatore) + 'Data_Y.csv'
                Y1 = np.concatenate((self.Y_b_in[contatore], np.ones((M, 1))), axis=1)
                np.savetxt(s, Y1, delimiter=',')

            # todo: salvo anche classi del classificatore
            # save classifier
            # for i in range(self.n_class):
            #     folder_name = self.folder_name + '/Class'
            #     folder_exists = os.path.exists(folder_name)
            #     if i == 0:
            #         if not folder_exists:
            #             os.makedirs(folder_name)
            #         else:
            #             os.remove(folder_name + '/Data_X_class' + str(i) + '.csv')
            #             os.remove(folder_name + '/Data_Y_class' + str(i) + '.csv')
            #     s = folder_name + '/Data_X_class' + str(i) + '.csv'
            #     np.savetxt(s, self.RBF_model.GP_class['X'][i], delimiter=',')
            #     s = folder_name + '/Data_Y_class' + str(i) + '.csv'
            #     np.savetxt(s, self.RBF_model.GP_class['Y'][i], delimiter=',')

    def savings(self):
        """
        If the boolean inside the class "save_experiment" is true, it saves the optimization.
        :return:
        """
        if self.save_experiment:
            if self.load_experiment_foldername is None:
                if self.folder_name is None or self.folder_name == "":
                    self.folder_name = "experimentsFolder/1Experiment_" + self.nameOpt + "_" + datetime.now().strftime(
                        "%Y%m%d-%H%M%S")
            else:
                self.folder_name = self.load_experiment_foldername

    """ Chooser functions """

    def _model_chooser(self, i):
        """
        Chose the type of model to create
        :return: instance of a model
        """
        appoggio = self.fvars_x_Ratio[i]
        appoggio1 = self.fvars_cat_Ratio[i]
        appoggio2 = self.fvars_not_cat_Ratio[i]

        # if i == 0:
        #     appoggioz = self.a0
        #     appoggio1z = self.fvars_cat0
        #     appoggio2z= self.fvars_not_cat0
        # else:
        #     appoggioz = self.a1
        #     appoggio1z = self.fvars_cat0
        #     appoggio2z = self.fvars_not_cat0

        if self.model_type == 'RBF':
            X = self.kwargs.get("X", [])
            b_in = self.kwargs.get("b_in", None)
            b_eq = self.kwargs.get("b_eq", None)
            beta = self.kwargs.get("beta", None)
            normalize_X = self.kwargs.get("normalize_X", True)
            epslon = self.kwargs.get("epslon", 0.1)
            lam = self.kwargs.get("lam", 0.00001)
            sigma = self.kwargs.get("sigma", 0.01)
            print("Sigma: ", sigma)
            GP_class = self.kwargs.get('GP_class', None)
            RBF_function_name = self.kwargs.get('RBF_function_name', "Gaussian")
            # todo: qui verrà fatto i:i*2 o cose simili
            return RBFModelFinale(appoggio, appoggio1, appoggio2, X=X,
                                  normalize_X=normalize_X,
                                  beta=beta, b_in=b_in, b_eq=b_eq,
                                  epslon=epslon,
                                  sigma=sigma, lam=lam, GP_class=GP_class, RBF_function_name=RBF_function_name)
        else:
            raise NotImplementedError(f'Model {self.model_type} not supported')

    def __anchor_points_chooser(self):
        """
        Choose the anchor points generator
        :return: The anchor points generator chosen
        """
        if self.evaluator_type == 'sequential_batch':
            return RandomAnchorPointsGenerator(self.fvars_x, self.acquisition.get_acquisition, self.normalize_X,
                                               self.anchor_points_samples)
            # return RandomAnchorPointsGenerator(self.fvars_x, self.IDW_acquisition, self.normalize_X,self.anchor_points_samples)
        # elif self.evaluator_type == 'thompson':
        #     # todo: thompson anchor non implementato
        #     return ThompsonAnchorPointsGenerator(self.fvars_x, self.predict, self.normalize_X,
        #                                          self.anchor_points_samples)
        else:
            raise NotImplementedError(f'Evaluator {self.evaluator_type} not supported')

    def _evaluator_chooser(self):
        """
        Choose the evaluator
        :return: The chosen evaluator
        """
        if self.evaluator_type == 'sequential_batch':
            return BatchSequential(self.acquisition, self.batch_size, self.fvars_x, self.normalize_X, None, None,
                                   self.acquisition_optimizer)
            # return BatchSequential(self.IDW_acquisition, self.batch_size, self.fvars_x,self.normalize_X, None, None,self.acquisition_optimizer)
        # elif self.evaluator_type == 'thompson': # TODO thompson eval non implementato
        #     return ThompsonBatch(self.acquisition, self.batch_size, self.fvars_x,
        #                          self.normalize_X,
        #                          self.one_hot_permutations, self.acquisition_optimizer)
        else:
            raise NotImplementedError(f"Evaluator {self.evaluator_type} not supported")

    def __acquisition_optimizer_chooser(self):
        """
        Choose the acquisition optimizer
        :return: The acquisition optimizer
        """
        if self.acquisition_optimizer_type == 'lbfgsb':
            return LBFGSB(self.bnd_all)
        # todo: test other optimizators
        elif self.acquisition_optimizer_type == 'de':
            return DE()
        elif self.acquisition_optimizer_type == 'slsqp':
            return SLSQP()
        elif self.acquisition_optimizer_type == 'powell':
            return Powell()
        elif self.acquisition_optimizer_type == 'nelder-mead':
            return NelderMead()
        else:
            raise NotImplementedError(f"Optimizer {self.acquisition_optimizer_type} not supported")

    def __acquisition_chooser(self):
        """
        Choose the acquisition function
        :return: Acquisition function chosen
        """
        # Multiple Ys only supported in Preferences
        if self.acquisition_type != 'Preferences' and len(self.fvars_y) != 1:
            raise ValueError("Multi output only supported with Preferences")

        if self.acquisition_type == 'Preferences':
            # identity function if none and id there is only 1 output
            if self.g is None:
                def g(X, Y, Model):
                    return np.sum([Y[:, i] for i in range(len(self.fvars_y))], axis=0)

                self.g = g
            return Preferences(self, N=0, jitter=0)
        else:
            raise NotImplementedError(f"Acquisition {self.acquisition_type} not supported")

    def _get_rounded_opt(self, x):
        """
        Used to round x_next
        :param x: float to be rounded
        :return: rounded float
        """
        X_round = np.zeros_like(x)
        idx = 0
        for fvar in self.fvars_x:
            dim = fvar.get_dim()
            X_round[:, idx:idx + dim] = fvar.get_rounded(x[:, idx:idx + dim])
            idx = idx + dim
        return X_round

    def processGaussian(self):
        """
        #todo: gaussian process
        :return:
        """
        for m in self.models:
            X_list = copy.copy(m.GP_class['X'])
            Y_list = copy.copy(m.GP_class['Y'])
            # self.Y_pref[i] = self.express_class_preference(f = self.my_fun_class[i])
            for i in range(len(X_list)):
                X_list[i] = np.concatenate((X_list[i], self.X_next))
                y_class = int(self.express_class_preference(self.my_fun_class[i], self.X_next, **self.kwargs))
                print('---------------------------------------------------------------------------')
                print(Y_list[i])
                print(y_class)
                print('---------------------------------------------------------------------------')
                Y_list[i] = np.concatenate((Y_list[i], [y_class]))
            m.GP_class['X'] = X_list
            m.GP_class['Y'] = Y_list

    def generate_anchor_points(self, restricted_domains_indexes=None, restricted_domains_values=None, num_anchor=5,
                               anchor_points_samples=None, norm=False, selective_percentage=0,
                               exploration=True):

        # if restricted_domains_indexes is not None and restricted_domains_values is not None:
        #     new_dict = copy.deepcopy(self.x_anchor)
        #     idx_fixed = 0
        #     for i in restricted_domains_indexes:
        #         new_dict[i]['domain'] = restricted_domains_values[idx_fixed]
        #         idx_fixed += 1
        #     self.anchor_points_generator.set_restricted_domain(self.init_variables2(new_dict))

        anchor_points = self.anchor_points_generator.get(num_anchor=num_anchor,
                                                         anchor_points_samples=anchor_points_samples, norm=norm,
                                                         selective_percentage=selective_percentage)

        self.anchor_points_generator.set_restricted_domain(None)
        return anchor_points

    # def init_variables2(self, fvars):
    #     variables_list = []
    #     for function_var in fvars:
    #         if function_var['type'] == 'continuous':
    #             variables_list.append(ContinuousVariable(function_var['name'], np.array(function_var['domain'])))
    #         elif function_var['type'] == 'discrete':
    #             variables_list.append(DiscreteVariable(function_var['name'], np.array(function_var['domain'])))
    #         elif function_var['type'] == 'categorical':
    #             variables_list.append(CategoricalVariable(function_var['name'], np.array(function_var['domain'])))
    #         else:
    #             raise Exception('Variable type not recognized:' + function_var['type'])
    #     return variables_list

    """ Model Update"""

    def __update_model(self, model, idx):
        """
        In this phase, each model will take their correct b_in, b_eq and ind_best and will update itself.
        :param model: model (each model present in self.models)
        :param idx: index for this model
        :return:
        """

        x_in_model = self.X.copy()[:, idx * self.ratio: (idx + 1) * self.ratio].reshape(
            (self.X.copy().shape[0], self.ratio))
        # x_in_model = self.X
        # v = 1 - idx
        # x_in_model[:, v] = 0
        y_b_in_model = self.Y_b_in[idx]
        y_b_eq_model = self.Y_b_eq[idx]
        model.update_model(x_in_model, bin=y_b_in_model, beq=y_b_eq_model, kfold=self.kfold)

    def update(self, exploration=True):
        """
        Used to update all models.
        :param exploration: before updating models, it's possible setting the exploration parameter. If false, the algoritm will take next point in exploitative way.
        :return:
        """
        self.acquisition.set_exploration(exploration)
        for idx, model in enumerate(self.models):
            self.__update_model(model, idx)

    """ Exposed functions """

    def add_evaluations(self, x_next, eval):
        """
        Used to add x_next comparison for each model.
        :param x_next: point compared to the best of each model
        :param eval: array of comparisons (contains -1 if x_next is better of the best so far, 1 otherwise)
        :return:
        """
        (N, n) = np.shape(self.X)
        self.X = np.concatenate((self.X, x_next))
        self.X_all = np.concatenate((self.X_all, self.X_next_all))
        Y_pref = np.zeros(self.batch_size)
        contatore = 0
        for indiceModello in self.models:
            value = eval[contatore]
            for i in range(self.batch_size):
                Y_pref[i] = value[i]
                if Y_pref[i] == 1:
                    self.Y_b_in[contatore] = np.concatenate(
                        (self.Y_b_in[contatore], np.array([[self.Y_ind_best[contatore], N + i]])))
                else:
                    if Y_pref[i] == 0:
                        self.Y_b_eq[contatore] = np.concatenate(
                            (self.Y_b_eq[contatore], np.array([[N + i, self.Y_ind_best[contatore]]])))
                    else:
                        self.Y_b_in[contatore] = np.concatenate(
                            (self.Y_b_in[contatore], np.array([[N + i, self.Y_ind_best[contatore]]])))
                    self.Y_ind_best[contatore] = N + i
            contatore += 1

        # todo: modello gp update
        if self.n_class > 0:
            self.processGaussian()
        self._save_experiment()
        self.n_iteration += 1

    def run_optimization(self, n_in=5, exploration=True):
        restricted_domain = None  # [0, 1, 2, 3, 4, 5, 6, 7]
        bounds = None
        """
        Core of the program.
        1. update of all RBF models
        2. generation of 1000 anchorpoints
        3. evaluation of anchorpoints by acquisition function
        4. optimization of the best "n_in" anchorpoints by optimizer function (default LBFGSB)
        :param n_in: number of anchor points to optimize
        :param exploration: is true optimization will search in explorative way, otherwise in exploitative
        :return:
        """
        self.update(exploration=exploration)
        n = np.shape(self.X)[1]
        self.X_batch = np.zeros((self.batch_size, n))
        for ind_batch in range(self.batch_size):
            x0 = self.generate_anchor_points(restricted_domains_indexes=restricted_domain,
                                             restricted_domains_values=bounds, num_anchor=n_in, selective_percentage=0,
                                             norm=self.normalize_X)
            x_next = self.evaluator.compute_batch(restricted_domains_indexes=restricted_domain,
                                                  restricted_domains_values=bounds, anchor_points=x0, maxiter=15000,
                                                  epsilon=1e-08)
            self.X_batch[ind_batch, :] = x_next
        (N, n) = self.getShape()
        self.X_next_all = np.zeros((self.batch_size, n))
        for ind_n in range(self.batch_size):
            self.X_next_all[ind_n:ind_n + 1, :] = np.reshape(self.X_batch[ind_n, :], (-1, n))
        if self.normalize_X:
            self.X_next = PreferenceOptimization3.utils.math.denormalize_X(self.fvars_x, self.X_next_all)
        self.X_next = self._get_rounded_opt(self.X_next)
        self.savings()
        return self.X_next

    def getShape(self):
        return np.shape(self.X)

    def run_optimizationTRY(self, n_in=5, exploration=True, er=None):
        self.update(exploration=exploration)
        n = self.getShape()[1]
        self.X_batch = np.zeros((self.batch_size, n))
        for ind_batch in range(self.batch_size):
            self.X_batch[ind_batch, :] = er
        (N, n) = self.getShape()
        self.X_next_all = np.zeros((self.batch_size, n))
        for ind_n in range(self.batch_size):
            self.X_next_all[ind_n:ind_n + 1, :] = np.reshape(self.X_batch[ind_n, :], (-1, n))
        # if self.normalize_X:
        self.X_next = self.X_next_all
        self.X_next = self._get_rounded_opt(self.X_next)
        self.savings()
        return self.X_next

    def normalize(self, x):
        if len(x) == 1:
            return x
        min = np.min(x)
        max = np.max(x)
        return (x - min) / (max - min)

    def predict_2(self, x):
        return self.models[0].predict(x)

    def predict(self, x):
        (N, n) = self.getShape()
        x = np.reshape(x, (-1, n))
        xd = x  # denormalize_X(self.fvars_x, x)
        nx = np.shape(x)[0]

        f_hat_sum = np.zeros(len(xd))
        f_hat_X_sum = np.zeros(len(self.X))

        a1 = []
        a2 = []

        bestYY = []
        index = 0
        indiceModello = 0
        for m in self.models:
            xYY = m.X[self.Y_ind_best[index]]
            f_hat_X = m.predict(m.X)
            mm = np.min(f_hat_X)
            MM = np.max(f_hat_X)
            Delta_F = MM - mm
            f_hat = m.predict(xd[:, indiceModello * self.ratio:(indiceModello + 1) * self.ratio])
            YYY = (m.predict(xYY) - mm) / Delta_F
            bestYY.append(YYY)
            f_hat = (f_hat - mm) / Delta_F
            a1.append(self.normalize(f_hat_X))
            a2.append(f_hat)
            index += 1
            indiceModello += 1

        f_hat_X_sum = self.g(self.X, np.array(a1).T, self)
        f_hat_sum = self.g(self.X, np.array(a2).T, self)

        mm = np.min(f_hat_X_sum)
        MM = np.max(f_hat_X_sum)
        Delta_F = MM - mm
        a = 1 * (f_hat_sum - mm) / Delta_F
        # if self.init == 1:
        #     for i in range(len(self.X_batch) - 1):
        #         d = (LA.norm(x - self.X_batch[i, :]) ** 2) / (1 ** 2)
        #         a = a * (1 - np.exp(-d))

        return a

    def predict_no_norm(self, x):
        a = self.models[0].predict(x)
        return a

    def _generate_samples_Try(self, n_iter=10):
        x_init = np.empty((n_iter, 0))
        for function_var in self.fvars_x:
            if function_var.get_type() == 'continuous':
                x = np.random.uniform(function_var.get_domain()[0], function_var.get_domain()[1],
                                      (n_iter, 1))
            elif function_var.get_type() == 'discrete':
                x = np.random.choice(function_var.get_domain(), (n_iter, 1))
            elif function_var.get_type() == 'categorical':
                x = function_var.get_one_hot(np.random.choice(function_var.get_domain(), (n_iter, 1)))
            else:
                raise ValueError('Variable type not supported')
            x_init = np.column_stack((x_init, x))
        return x_init

    def compute_preference(self, f, x1, x2):
        f_1 = f(x1)
        f_2 = f(x2)
        N_val = np.shape(x1)[0]
        pref_hat = np.zeros(N_val)
        for ind in range(N_val):
            if f_1[ind] <= f_2[ind]:
                pref_hat[ind] = 1
            else:
                pref_hat[ind] = -1
        return pref_hat

    def predict_preference(self, x1, x2):
        f_hat1 = self.predict(x1)
        f_hat2 = self.predict(x2)
        N_val = np.shape(x1)[0]
        pref_hat = np.zeros(N_val)
        for ind in range(N_val):
            if f_hat1[ind] <= f_hat2[ind]:
                pref_hat[ind] = 1
            else:
                pref_hat[ind] = -1
        return pref_hat

    def get_Reliability(self, n=100):
        a = self._generate_samples_Try(n_iter=n)
        b = self._generate_samples_Try(n_iter=n)
        pref_true = self.compute_preference(self.objectives[0], a, b)
        pref_hat = self.predict_preference(a, b)
        arr = pref_true - pref_hat
        # print(pref_true - pref_hat)
        rel = (n - np.count_nonzero(arr)) / n * 100
        print(rel, "%")
        return a, b, arr, rel

    def __g_function_chooser(self, g):
        if g is None:
            def g(X, Y, Model):
                return np.sum([Y[:, i] for i in range(len(self.fvars_y))], axis=0)
            return g
        else:
            return g
