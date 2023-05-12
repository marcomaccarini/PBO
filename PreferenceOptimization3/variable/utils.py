import numpy as np

from PreferenceOptimization3.variable.continuous_variable import ContinuousVariable
from PreferenceOptimization3.variable.discrete_variable import DiscreteVariable
from PreferenceOptimization3.variable.categorical_variable import CategoricalVariable


def init_variables(fvars):
    variables_list = []
    for function_var in fvars:
        if function_var['type'] == 'continuous':
            variables_list.append(ContinuousVariable(function_var['name'], np.array(function_var['domain'])))
        elif function_var['type'] == 'discrete':
            variables_list.append(DiscreteVariable(function_var['name'], np.array(function_var['domain'])))
        elif function_var['type'] == 'categorical':
            variables_list.append(CategoricalVariable(function_var['name'], np.array(function_var['domain'])))
        else:
            raise Exception('Variable type not recognized:' + function_var['type'])
    return variables_list


def split_variables(fvars):
    # Separate categorical from others
    fvars_cat = []
    fvars_not_cat = []
    for fvar in fvars:
        if fvar.get_type() == 'categorical':
            fvars_cat.append(fvar)
        else:
            fvars_not_cat.append(fvar)
    return fvars_cat, fvars_not_cat