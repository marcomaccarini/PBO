import numpy as np
import NLP_Opt as NLP

path_for_savings = "notebooks/"
description = ""  # "200Points"


def p1(o11, o12):
    return (o11 + 2) ** 2 + (o12 - 1) ** 2


def p2(o21, o22):
    return (o21 - 3) ** 2 + (o22 - 0.5) ** 2


def alp(al1, al2):
    return 20 * (1 + (al1 - 0.5) ** 2 + (al2 - 0.5) ** 2)


def objective_function_original(x):
    o11 = x[2]
    o12 = x[3]
    o21 = x[4]
    o22 = x[5]
    P1 = p1(o11, o12)
    P2 = p2(o21, o22)
    obj = P1 + P2
    return obj


def calculate_t_bar_original(x):
    al1 = x[0]
    al2 = x[1]
    o11 = x[2]
    o12 = x[3]
    o21 = x[4]
    o22 = x[5]
    P1 = p1(o11, o12)
    P2 = p2(o21, o22)
    Alpha = alp(al1, al2)
    return Alpha / (1 + P1 + P2)


point_for_T_bar = [0, 0, 0, 0, 0, 0]
T_bar = calculate_t_bar_original(point_for_T_bar)


def constraint_original(x):
    # t < t_bar
    al1 = x[0]
    al2 = x[1]
    o11 = x[2]
    o12 = x[3]
    o21 = x[4]
    o22 = x[5]
    P1 = p1(o11, o12)
    P2 = p2(o21, o22)
    Alpha = alp(al1, al2)
    T = Alpha / (1 + P1 + P2)
    return T_bar - T


buond_alpha = 2


def c1_L(x):
    return buond_alpha - x[0]


def c1_U(x):
    return buond_alpha + x[0]


def c2_L(x):
    return buond_alpha - x[1]


def c2_U(x):
    return buond_alpha + x[1]


buond_phase = 10


def c3_L(x):
    return buond_phase - x[2]


def c3_U(x):
    return buond_phase + x[2]


def c4_L(x):
    return buond_phase - x[3]


def c4_U(x):
    return buond_phase + x[3]


def c5_L(x):
    return buond_phase - x[4]


def c5_U(x):
    return buond_phase + x[4]


def c6_L(x):
    return buond_phase - x[5]


def c6_U(x):
    return buond_phase + x[5]


import dill

OPT_phase_1 = dill.load(open(path_for_savings + "OPT_phase_1" + description + ".pkl", "rb"))
OPT_phase_2 = dill.load(open(path_for_savings + "OPT_phase_2" + description + ".pkl", "rb"))
OPT_transient_1_2 = dill.load(open(path_for_savings + "OPT_transient_1_2" + description + ".pkl", "rb"))

# In[92]:


ist = NLP.NLP_OPT(objective_function_original,
                  [  # constraint_original,
                      c1_L, c1_U, c2_L, c2_U, c3_L, c3_U, c4_L, c4_U, c5_L, c5_U, c6_L, c6_U, ],
                  OPT_transient_1_2.fvars_x)
# ist = NLP.NLP_OPT(calculate_t_bar_original,
#                  [c1_L,c1_U,c2_L,c2_U,c3_L,c3_U,c4_L,c4_U,c5_L,c5_U,c6_L,c6_U,],
#                  OPT_transient_1_2.fvars_x)


# In[93]:


result_original, al_res_original = ist.solve(100)
print("solution=            ", result_original)
print("ob(solution)=        ", objective_function_original(result_original))
print("contraint(solution)= ", calculate_t_bar_original(result_original))


# In[ ]:


# ## 2. minimizzo le surrogate

# In[94]:


def objective_function_surrogate(x):
    al1 = x[0]
    al2 = x[1]
    o11 = x[2]
    o12 = x[3]
    o21 = x[4]
    o22 = x[5]
    P1 = OPT_phase_1.predict_no_norm([o11, o12])
    P2 = OPT_phase_2.predict_no_norm([o21, o22])
    return P1 + P2


# ### 2.1. $\bar{T}_{1,2}=?$

# In[95]:


def calculate_t_bar_surrogate(x):
    al1 = x[0]
    al2 = x[1]
    o11 = x[2]
    o12 = x[3]
    o21 = x[4]
    o22 = x[5]
    # P1 = OPT_phase_1.predict([o11,o12])
    # P2 =  OPT_phase_2.predict([o21,o22])
    Alpha = OPT_transient_1_2.predict_no_norm([al1, al2, o11, o12, o21, o22])
    # P1 = p1(o11,o12)
    # P2 = p2(o21,o22)
    # Alpha = alp(al1,al2)
    # T = Alpha/(1+P1+P2)
    T = Alpha
    return T


# In[96]:


T_bar_surrogate = calculate_t_bar_surrogate(point_for_T_bar)[0]  # -0.0128
# T_bar_surrogate = 0.2437


# In[97]:


T_bar_surrogate = T_bar_surrogate - T_bar_surrogate * 0.0498

# In[98]:


T_bar_surrogate


def constraint_surrogate(x):
    # t < t_bar
    al1 = x[0]
    al2 = x[1]
    o11 = x[2]
    o12 = x[3]
    o21 = x[4]
    o22 = x[5]
    T = OPT_transient_1_2.predict_no_norm([al1, al2, o11, o12, o21, o22])
    # P1 = p1(o11,o12)
    # P2 = p2(o21,o22)
    # Alpha = alp(al1,al2)
    # T = Alpha/(1+P1+P2)
    return T_bar_surrogate - T


ist = NLP.NLP_OPT(objective_function_surrogate,
                  [  # constraint_surrogate,
                      c1_L, c1_U, c2_L, c2_U, c3_L, c3_U, c4_L, c4_U, c5_L, c5_U, c6_L, c6_U, ],
                  # [],
                  OPT_transient_1_2.fvars_x)

# In[101]:


result_surrogate, all_res_surrogate = ist.solve(5)

# In[102]:


print(all_res_surrogate)

# In[107]:


np.set_printoptions(suppress=True)
print("result surrogate =     ", result_surrogate)
print()
print("*** Surrogate ***")
print("objective_function_surrogate(solution)=        ", objective_function_surrogate(result_surrogate))
print("constraint_surrogate(solution)= ", ((calculate_t_bar_surrogate(result_surrogate))))
print()

print("*** Original ***")
print("ob(solution)=        ", objective_function_original(result_surrogate))
print("contraint(solution)= ", calculate_t_bar_original(result_surrogate))

# In[108]:


print(result_original)

# In[109]:


for ew in all_res_surrogate:
    print(calculate_t_bar_original(ew))

# In[ ]:


# In[ ]:
