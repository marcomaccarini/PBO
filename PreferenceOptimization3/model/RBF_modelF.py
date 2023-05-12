from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import cvxpy as cp
from sklearn.model_selection import KFold
import copy
import PreferenceOptimization3.utils.math
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF  # ,  WhiteKernel
from sklearn.preprocessing import normalize
from PreferenceOptimization3.RBF_funcitons.Gaussian import Gaussian
from PreferenceOptimization3.RBF_funcitons.InverseQuadratic import InverseQuadratic
from PreferenceOptimization3.RBF_funcitons.Multiquadratic import Multiquadratic
from PreferenceOptimization3.RBF_funcitons.Linear import Linear
from PreferenceOptimization3.RBF_funcitons.ThinPlateSpline import ThinPlateSpline
from PreferenceOptimization3.RBF_funcitons.InverseMultiQuadratic import InverseMultiQuadratic


class RBFModelFinale(object):

    def __init__(self, fvars_x, fvars_cat, fvars_not_cat, X=[], b_in=None, b_eq=None, beta=None,
                 normalize_X=True, epslon=0.1, lam=0.00001, sigma=0.1, GP_class=None, RBF_function_name="Gaussian"):

        self.fvars_x = fvars_x

        self.fvars_cat = fvars_cat
        self.fvars_not_cat = fvars_not_cat

        self.X = X
        self.b_in = b_in
        self.b_eq = b_eq

        self.beta = beta
        self.epslon = epslon
        self.sigma = sigma
        self.lam = lam
        self.eps_in = None
        self.eps_eq = None

        self.normalize_X = normalize_X

        self.GP_class = GP_class
        self.score = 0

        self.GP_loss = None
        self.RBF_function_name = RBF_function_name
        self.RBF_function = self.__rbf_function_chooser()

    def __rbf_function_chooser(self):
        if self.RBF_function_name == "Gaussian":
            return Gaussian()
        elif self.RBF_function_name == "Inverse-Quadratic":
            return InverseQuadratic()
        elif self.RBF_function_name == "Multiquadratic":
            return Multiquadratic()
        elif self.RBF_function_name == "Linear":
            return Linear()
        elif self.RBF_function_name == "Thin-Plate-Spline":
            return ThinPlateSpline()
        elif self.RBF_function_name == "Inverse-Multiquadratic":
            return InverseMultiQuadratic
        else:
            raise Exception('RBF function not recognized')

    def RBF(self, x):
        """
        Used to calculate x's rbf-score
        # todo: metto appunto della pagina paper in cui spiego
        :param x: point to evaluate the score
        :return: array of the same length of X,
                    for each elem in X  calculates RBF(x,elem)
        """
        X = PreferenceOptimization3.utils.math.normalize_X(self.fvars_x, self.X)
        (N, nu) = np.shape(X)
        Phi_vect = np.zeros((N, 1))
        for i in range(N):
            # d = (self.epslon ** 2) * LA.norm(x - X[i, :]) ** 2
            # Phi_vect[i, 0] = np.exp(-d)
            d = LA.norm(x - X[i, :])
            Phi_vect[i, 0] = self.RBF_function.rbf(self.epslon, d)
        return Phi_vect

    def get_beta(self, x, b_in, b_eq, c_in=None, c_eq=None):
        """
        used to calculate beta array
        :param x: list of point in the model
        :param b_in: each lines contain 2 tested point x1 and x2 where ∏(x1,x2) = -1
        :param b_eq: each lines contain 2 tested point x1 and x2 where ∏(x1,x2) = 0
        :param c_in: for future implementation
        :param c_eq: for future implementation
        :return:
        """
        self.X = x

        X = PreferenceOptimization3.utils.math.normalize_X(self.fvars_x, self.X)
        self.b_in = b_in
        self.b_eq = b_eq

        b_in = self.b_in.astype(int)
        b_eq = self.b_eq.astype(int)
        sigma = self.sigma
        lam = self.lam

        M_in = np.shape(b_in)[0]
        M_eq = np.shape(b_eq)[0]
        (N, nu) = np.shape(X)

        if c_in is None or len(c_in) == 0:
            c_in = np.array(M_in * [1])
            c_in = np.reshape(c_in, (1, -1))

        if c_eq is None or len(c_eq) == 0:
            c_eq = np.array(M_eq * [1])
            c_eq = np.reshape(c_eq, (1, -1))

        beta = cp.Variable(N)

        eps_in = cp.Variable(M_in)

        if M_eq >= 1:
            b_eq = b_eq.astype(int)
            eps_eq = cp.Variable(M_eq)
            obj = (lam / (2 * N)) * cp.sum_squares(beta) + c_in @ eps_in + c_eq @ eps_eq
            constraints1 = [eps_in >= 0, eps_eq >= 0]
        else:
            eps_eq = 0
            obj = lam / 2 * cp.sum_squares(beta) + c_in @ eps_in + 1 * cp.sum_squares(eps_in)
            constraints1 = [eps_in >= 0]

        objective = cp.Minimize(obj)

        constraints_in = []
        for ind in range(M_in):
            Phi0 = self.RBF(X[b_in[ind, 0]])
            Phi1 = self.RBF(X[b_in[ind, 1]])

            CC = [beta.T @ Phi0 - beta.T @ Phi1 <= - sigma + eps_in[ind]]
            constraints_in = constraints_in + CC

        constraints_eq = []
        for ind in range(M_eq):
            Phi0 = self.RBF(X[b_eq[ind, 0]])
            Phi1 = self.RBF(X[b_eq[ind, 1]])
            CC1 = [beta.T @ Phi0 - beta.T @ Phi1 <= sigma + eps_eq[ind]]
            CC2 = [beta.T @ Phi0 - beta.T @ Phi1 >= - sigma - eps_eq[ind]]
            constraints_in = constraints_in + CC1 + CC2

        constraints = constraints1 + constraints_in + constraints_eq
        prob = cp.Problem(objective, constraints)
        # solver = cp.ECOS
        try:
            prob.solve(verbose=False, eps_rel=0.00000001, max_iter=1000000)

            beta1 = beta.value
            # print(beta.value)
            # if beta.value is None:
            #     print("is none")

            eps_in1 = eps_in.value
            if M_eq >= 1:
                eps_eq1 = eps_eq.value
            else:
                eps_eq1 = []

            # self.beta = utils.math.scaleValues(beta1, 1)
            self.beta = beta1
            self.eps_in = eps_in1
            self.eps_eq = eps_eq1
        except Exception as exception:
            # input(prob.status)
            self.beta = np.zeros(N)
            # assert type(exception).__name__ == 'NameError'
            # assert exception.__class__.__name__ == 'NameError'
            # assert exception.__class__.__qualname__ == 'NameError'
            print("N-", end='')  # , self.X, "  ", "epslon: ", self.epslon, " lam: ", self.lam)
        # print(self.beta)
        # min = np.min(self.beta)
        # max = np.max(self.beta)

        # self.beta = (self.beta - min) / (max - min)
        # print(self.beta)

    def reorder(self, xd, a):
        """
        Used to reorder from the smaller to the bigger
        :param xd:
        :param a:
        :return:
        """
        ee2 = a.reshape(len(a), 1)
        fin2 = np.column_stack((xd, ee2))
        fin2 = fin2[fin2[:, 0].argsort(), :]
        return fin2[:, 0], fin2[:, 1]

    def update_model(self, x, bin, beq, c_in=None, c_eq=None, kfold=None):
        """
        used to calibrate hyperparameters of rbf and calculate beta array (see pag. of the paper)
        #todo: mettere pagina paper
        :param x: list of point in the model
        :param bin: each lines contain 2 tested point x1 and x2 where ∏(x1,x2) = -1
        :param beq: each lines contain 2 tested point x1 and x2 where ∏(x1,x2) = 0
        :param c_in:
        :param c_eq:
        :param kfold: number of fold, used to make test of hyper parameters
        :return:
        """

        if kfold is None:
            self.get_beta(x, bin, beq, c_in, c_eq)
        else:
            # print("KFOLD. with ", kfold, " fields: ", end='')
            self.X = x
            self.b_in = bin
            self.b_eq = beq

            self2 = copy.copy(self)
            score_mean = []
            nlam = 1  # 8
            lam_vec = np.array([1e-6])  # np.logspace(-7, 0, nlam)

            epslon_vec = np.array([.1, .5, 1, 2, 10])  # np.logspace(-2, 2, nepslon)
            nepslon = len(epslon_vec)

            lam_index = 0
            epslon_index = 0
            score_epslon = []
            times = 0
            bs = None
            best_lam = 0
            best_epslon = 0
            best_sc = 0
            exit1 = False
            while exit1 is not True:
                # print('Cross-validation started\n')
                score_mean = np.zeros((nlam, nepslon))
                beta_val = np.zeros((nlam, nepslon), dtype=object)
                cv = KFold(n_splits=kfold, random_state=10, shuffle=True)
                for ind_lam in range(nlam):
                    for ind_epslon in range(nepslon):
                        self2.lam = lam_vec[ind_lam]
                        self2.epslon = epslon_vec[ind_epslon]
                        score = []
                        for train_index, test_index in cv.split(self.b_in):
                            b_in_app = copy.copy(self.b_in[train_index, :])
                            # self2.b_eq = self.b_eq[train_index]
                            self2.get_beta(self.X, b_in_app, self.b_eq, c_in, c_eq)
                            # if not np.any(self2.beta):
                            # print("self2.beta è none..")

                            b_in_test = self.b_in[test_index, :]
                            N_test = np.shape(b_in_test)[0]
                            count = 0
                            for ind in range(N_test):
                                ind1 = b_in_test[ind, 0].astype(int)
                                ind2 = b_in_test[ind, 1].astype(int)
                                # print('index')
                                # print(ind1)
                                f1 = self2.predict(self2.X[ind1:ind1 + 1, :])
                                f2 = self2.predict(self2.X[ind2:ind2 + 1, :])
                                if f1 < f2:
                                    count = count + 1
                            if not np.any(self2.beta):
                                count = -100
                            score.append(count / N_test)
                        score = np.array(score)
                        score_mean[ind_lam, ind_epslon] = np.mean(score)  # np.array(score_meanL)
                        beta_val[ind_lam, ind_epslon] = copy.copy(self2)  # np.array(score_meanL)

                epslon_i_max = np.argmax(score_mean, axis=1)
                score_epslon = np.zeros(nlam)
                score_bebe = np.zeros(nlam, dtype=object)
                for indr in range(nlam):
                    score_epslon[indr] = score_mean[indr, epslon_i_max[indr]]
                    score_bebe[indr] = beta_val[indr, epslon_i_max[indr]]
                lam_index = np.argmax(score_epslon)
                epslon_index = epslon_i_max[lam_index]
                sc = score_epslon[lam_index]
                betone = score_bebe[lam_index]

                # print(beta_val[epslon_i_max][lam_index])
                if sc > best_sc:
                    best_lam = lam_vec[lam_index]
                    best_epslon = epslon_vec[epslon_index]
                    best_sc = sc
                    bs = betone.beta

                if sc < 0.7 and times < 1:
                    times += 1
                    # print(str(times), end='')
                    # nlam = 8*times
                    # lam_vec = np.logspace(-7, 0, nlam)
                    nepslon = 30  # * times
                    epslon_vec = np.linspace(.001, 10 * times, nepslon * times)
                    # if self2.beta is not None:
                    # print(str(sc) + " rifare #" + str(times) + " mean^2: " + str(np.mean(self2.beta ** 2)) + "  std: " + str(np.std(self2.beta)))
                else:
                    # if self2.beta is not None:
                    # print(str(sc) + " esco #" + " epslon: " + str(best_epslon) + "  mean^2: " + str(np.mean(self2.beta ** 2)) + "  std: " + str(np.std(self2.beta)))
                    exit1 = True
                    # print(str(round(sc * 100, 2)) + "%")

            self.lam = best_lam
            self.epslon = best_epslon
            self.score = best_sc

            self.get_beta(self.X, self.b_in, self.b_eq, c_in, c_eq)

            if not np.any(self.beta):
                # todo: certe volte quando calcola i beta con tutti i b_in genera un errore nella LP, quindi assegno il beta calcolato con metà dei b_in
                self.beta = bs

            if not np.any(self.beta):
                # todo: certe volte quando calcola i beta con tutti i b_in genera un errore nella LP, quindi assegno il beta calcolato con metà dei b_in
                print("NaN")

            # todo: implemention
            if self.GP_class is not None:
                print('GP Started')
                X_list = self.GP_class['X']
                Y_list = self.GP_class['Y']
                self.GP_loss = [[]] * len(X_list)
                for i in range(len(X_list)):
                    X_class = X_list[i]
                    Y_class = Y_list[i]

                    kernel_class = 1.0 * RBF(1.0)
                    self.GP_loss[i] = GaussianProcessClassifier(kernel=kernel_class,
                                                                random_state=0, n_restarts_optimizer=20).fit(X_class,
                                                                                                             Y_class)
                    self.GP_loss[i].score(X_class, Y_class)
                    # print(self.GP_loss[i].score(X_class, Y_class))
                print('GP Ended')
            # print('Classification Ended')

    def IDW(self, x):
        """
        Inverse weight distance,
        # todo: allego pagina del paper e spiego
        :param x:
        :return:
        """
        X = PreferenceOptimization3.utils.math.normalize_X(self.fvars_x, self.X)
        (N, n) = np.shape(X)
        x = np.reshape(x, (-1, n))
        nx = np.shape(x)[0]
        den = np.zeros(nx)
        z = np.zeros(nx)
        for ind in range(nx):
            xx = x[ind:ind + 1, :]
            if np.sum(np.sum(xx == X, axis=1) >= n) >= 1:  # se la x che provo appartiene ai samples
                z[ind] = 0
            else:
                e = LA.norm(xx - X, axis=1) ** 2  # sono i pesi calcolati come in beporad pag 3, (3a)
                # w = 1 / e
                # den[ind] = np.sum(w)
                # pippo = 1 / den[ind]
                # paperino = np.arctan(pippo)
                # de = 2 / np.pi
                # z[ind] = paperino * de
                z[ind] = np.arctan(1 / sum(1 / e)) * 2 / np.pi
            # M = np.max(z)
            # m = np.min(z)
            # delta = M-m
            # z = (z-m)/delta
        # z = utils.math.normalize_IDW(z, 10)
        return z

    def predict(self, x):
        """
        Used to calculate f_hat(x)
        :param x: point to tested
        :return: f_hat(x)
        """
        X = PreferenceOptimization3.utils.math.normalize_X(self.fvars_x, self.X)

        (N, nu) = np.shape(X)
        if self.normalize_X:
            x = np.reshape(x, (-1, nu))
            x = PreferenceOptimization3.utils.math.normalize_X(self.fvars_x, x)
        beta = copy.copy(self.beta)
        epslon = self.epslon
        nx = np.shape(x)[0]
        f_hat = np.zeros(nx)
        for ind in range(nx):
            xx = x[ind, :]
            Phi = self.RBF(xx)
            try:
                f_hat[ind] = np.dot(beta, Phi)
            except:
                # print('Warning: the estimated surrogate is not reliable (RBF_model package)')
                f_hat[ind] = 1000000
        return f_hat

    def predict_preference(self, x1, x2):
        """
        Used to calculate ∏(x1,x2)
        :param x1:
        :param x2:
        :return: ∏(x1,x2)
        """
        f_hat1 = self.predict(x1)
        f_hat2 = self.predict(x2)
        N_val = np.shape(x1)[0]
        mask = f_hat1 <= f_hat2
        pref_hat = np.zeros(N_val)
        for ind in range(N_val):
            if f_hat1[ind] <= f_hat2[ind]:
                pref_hat[ind] = 1
            else:
                pref_hat[ind] = -1
        return pref_hat

    def printStuff(self):
        """
        :return:
        """
        print("X value:")
        index = 0
        for x in self.X:
            print(index, " ", x)
            x += 1
        '''
                print('pred class 0:')
                print(self.GP_loss[0].predict_proba(self.X)[:,0:1])
                print('pred class 1:')
                print(self.GP_loss[0].predict_proba(self.X)[:,0:1])         
        '''

        # self.acquisition.set_constraints(self.constraints)
