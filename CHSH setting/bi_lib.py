import numpy as np
import itertools as it
import cvxpy as cp
import cdd

from scipy.optimize import minimize
from scipy.optimize import Bounds
from NPA_lib import generate_NPA_matrix, check_quantum_NPA
from numpy import random as rand
from tqdm import tqdm

### CHSH setting ###
dimA = 2
dimB = 2
dimX = 2
dimY = 2

### All NS inequalities ###
positivity_NS = []
for a,b,x,y in it.product(*[range(2)]*4):
    pos_ineq = np.zeros((3,3))
    pos_ineq[0,0] = 1
    pos_ineq[x+1,0] = (-1)**a
    pos_ineq[0,y+1] = (-1)**b
    pos_ineq[x+1,y+1] = (-1)**(a+b)
    positivity_NS.append(pos_ineq)

### All CHSH inequalities ###
with open('bi_CHSH_ineqs.npy', 'rb') as f:
    CHSH_ineqs = np.load(f, allow_pickle=True)

######### NS extremal points #########
NS_H_rep = []
for ineq in positivity_NS:
    NS_H_rep.append(ineq.flatten())

NS_mat = cdd.Matrix(NS_H_rep, number_type='float')
NS_mat.rep_type = cdd.RepType.INEQUALITY
NS_poly = cdd.Polyhedron(NS_mat)
NS_V_rep = np.array(NS_poly.get_generators())
######################################

def bi_regularise_to_Q(P_raw, P_in, NPA_level=3, show_distance=False):
    '''
    A function to regularise the observed statistics P to the p which maximises the likelihood function.
    The method is discussed in [Physical Review A, 97(3):032309 (2018)].
    
    Inputs - P_raw: the observed statistics, numpy array P_raw[a,b,x,y] = P(a,b|x,y)
           - P_in: the observed input distribution, P_in[x,y] = p(x,y)
           - NPA_level: the level of the NPA hierarchy to use
           - show_distance: whether to print out the optimal distance between P and the closest p
    Output - p: the closest probability distribution in the quantum set
                numpy array p[a,b,x,y] = p(a,b|x,y)
    
    This function requires convex optimisation.
    We are using the library 'cvxpy' and the solver 'MOSEK'.
    '''
    dim_tot = dimA*dimB*dimX*dimY
    dim_range_tot = (range(dimA),range(dimB),range(dimX),range(dimY))

    ## OPTIMISATION VARIABLES
    pd = [cp.Variable(nonneg=True) for i in range(dim_tot)]
    pd = np.reshape(pd, (dimA,dimB,dimX,dimY))

    ## OBJECTIVE FUNCTION - KL Divergence between P_raw and pd
    object_func =  sum([-P_in[x,y]*P_raw[a,b,x,y]*cp.log(pd[a,b,x,y])/np.log(2)
                        for a,b,x,y in it.product(*dim_range_tot)])

    ## OPTIMISATION CONSTRAINTS
    constraints = []

    # Normalisation constraints
    pd_xy = np.sum(pd, axis=(0,1))
    for x,y in it.product(range(dimX),range(dimY)):
        constraints.append(pd_xy[x,y] == 1)

    # No-signalling
    for b,y in it.product(range(dimB),range(dimY)):
        no_sig_A2B = []
        for x in range(dimX):
            marg_prog = sum([pd[a,b,x,y] for a in range(dimA)])
            no_sig_A2B.append(marg_prog)
        for x in range(1,dimX):
            constraints.append(no_sig_A2B[0] == no_sig_A2B[x])
    for a,x in it.product(range(dimA),range(dimX)):
        no_sig_B2A = []
        for y in range(dimY):
            marg_prog = sum([pd[a,b,x,y] for b in range(dimB)])
            no_sig_B2A.append(marg_prog)
        for y in range(1,dimY):
            constraints.append(no_sig_B2A[0] == no_sig_B2A[y])

    # NPA constraint
    NPA_matrix = generate_NPA_matrix(pd, level=NPA_level)
    constraints.append(NPA_matrix >> 0)

    ## SOLVE OPTIMISATION
    prob = cp.Problem(cp.Minimize(object_func), constraints)
    prob.solve(verbose=False, solver='MOSEK')
    
    P_reg_ML = np.zeros((2,2,2,2))
    for a,b,x,y in it.product(*dim_range_tot):
        P_reg_ML[a,b,x,y] = pd[a,b,x,y].value

    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')
    if show_distance:
        KL_div = prob.value + sum([P_in[x,y]*P_raw[a,b,x,y]*np.log2(P_raw[a,b,x,y]) for a,b,x,y in it.product(*dim_range_tot)])
        print("KL-Divergence between the original and the regularised data :", KL_div)

    return P_reg_ML

def p2E_bi(p):
    '''
    Function to convert the conditional probability 'p' to its correlator representation E.

    Input - p: tripartite conditional probability, numpy array p[a,b,x,y] = p(a,b|x,y)
    Output - Evector: correlators, numpy array E[x,y] = <A_(x+1)B_(y+1)> where A_0,B_0 are the identity.
    '''
    Evector = np.zeros((3,3))
    
    dims = np.shape(p)
    dimA = dims[:2]
    dimQ = dims[2:]
    dimE = [dim+1 for dim in dimQ]
    
    subindices = [range(dim+1) for dim in dimQ]
    for x,y in it.product(*subindices):
        if x==0 and y==0: 
            Evector[x,y] = np.sum(p, axis=(0,1))[0,0]
            continue
            
        q_indices = [x,y]
        not_in = []
        yes_in = []
        for i in range(2):
            if q_indices[i]==0: not_in.append(i)
            else: yes_in.append(i)
                
        if len(yes_in)==1:
            sum_axis = tuple(not_in+[q+2 for q in not_in])
            p_summed = np.sum(p, axis=sum_axis)/2
            Evector[x,y] = sum([(-1)**a*p_summed[a,q_indices[yes_in[0]]-1] for a in range(2)])
        elif len(yes_in)==2:
            Evector[x,y] = sum([(-1)**(a+b)*p[a,b,x-1,y-1] for a,b in it.product(range(2),repeat=2)])
        else:
            raise Exception("There occured some problem converting the probability distribution to correlators.")
    return Evector

def E2p_bi(E, p=0, cvxpy=False):
    '''
    Function to convert the correlators 'Evector' to its probability representation.
    
    Input - Evector: correlators, numpy array E[x,y] = <A_(x+1)B_(y+1)>
    Output - p: conditional probability, numpy array p[a,b,x,y] = p(a,b|x,y)
    '''
    if not cvxpy:
        p = np.zeros((2,2,2,2))
    dimQ = [dim-1 for dim in np.shape(E)]
    subindices = [range(dim) for dim in dimQ]
    for a,b in it.product(*[range(2)]*2):
        for x,y in it.product(*subindices):
            p[a,b,x,y] = (1+(-1)**a*E[x+1,0]+(-1)**b*E[0,y+1]+(-1)**(a+b)*E[x+1,y+1])/4
    return p

def norm(P1, P2):
    if not len(P1)==len(P2):
        print("The two vectors must have the same length.")
    else:
        value = sum([abs(P1[i]-P2[i]) for i in range(len(P1))])/2 # Total variation distance
        # value = np.sqrt(sum([(P1[i]-P2[i])**2 for i in range(len(P1))])) # 2-norm
        return value

def GP_bi_2out_fullS_GivenH(E_data, given_ineqs, xbar, ybar, show=False):
    '''
    Solving optimisation calculating the guessing probability of 2 outputs for given inputs 'xbar, ybar'
    AGAINST adversaries characterised by 'given ineqs' in the bipartite setting using the full statistics.
    
    Inputs - E: the observed statistics in the correlator representation
                numpy array E[x,y] = <A_(x+1)B_(y+1)> with A_0=B_0=I
           - xbar: the fixed 'x'
           - ybar: the fixed 'y'
           - show: whether to print out the details of the optimisation
    Output - prob.value: the optimal guessing probability
           - opt_Evectors: the optimal solutions (optimal strategies for the adversary)
           
    This function requires convex optimisation. We use the library 'cvxpy' with solver 'MOSEK'
    '''
    ## Fixed parameters
    dim_E = (dimX+1)*(dimY+1)

    ## Optimisation Variables
    # 1. correlators
    E = []
    for e1,e2 in it.product(range(dimA),range(dimB)):
        E_e = [cp.Variable() for i in range(dim_E)]
        E_e = np.reshape(E_e, (dimX+1,dimY+1))
        E.append(E_e)
    E = np.reshape(E, (dimA,dimB,dimX+1,dimY+1))

    # 2. normalisations
    norm = [cp.Variable(nonneg=True) for e in range(dimA*dimB)] # norm of each p_e
    norm = np.reshape(norm, (dimA,dimB))

    ## Objective Function for x=xbar, y=ybar
    elements = []
    for a,b in it.product(range(dimA),range(dimB)):
        pd = (E[a][b][0,0] + (-1)**a*E[a][b][xbar+1,0] + (-1)**b*E[a][b][0,ybar+1] + (-1)**(a+b)*E[a][b][xbar+1,ybar+1])/4
        elements.append(pd)
    object_func = sum(elements)

    ## Optimisation Constraints
    constraints = []

    # Observed statistics constraints
    avg_E = np.sum(E, axis=(0,1))
    for x,y in it.product(range(dimX+1),range(dimY+1)):
        constraints.append(avg_E[x,y] == E_data[x,y])

    # Normalisation conditions
    for e1,e2 in it.product(range(dimA),range(dimB)):
        constraints.append(norm[e1,e2] <= 1)
        for x,y in it.product(range(dimX+1),range(dimY+1)):
            if x==0 and y==0:
                constraints.append(E[e1,e2][0,0] == norm[e1,e2]) # Constant part
            constraints.append(E[e1,e2][x,y] <= norm[e1,e2])
            constraints.append(E[e1,e2][x,y] >= -norm[e1,e2])
    constraints.append(np.sum(norm) == 1) # They should sum up to 1.

    for e1,e2 in it.product(range(dimA),range(dimB)):
        for i in range(len(given_ineqs)):
            value = given_ineqs[i].flatten()@E[e1][e2].flatten()
            constraints.append(value >= 0)

    ## Solve the Optimisation
    prob = cp.Problem(cp.Maximize(object_func), constraints)
    prob.solve(verbose=show, solver='MOSEK')
    
    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')
    
    ## Extract the optimal solution
    opt_E = []
    for e1,e2 in it.product(range(dimA),range(dimB)):
        opt_E_e = np.zeros((dimX+1, dimY+1))
        for x,y in it.product(range(dimX+1),range(dimY+1)):
            opt_E_e[x,y] = E[e1][e2][x,y].value/norm[e1,e2].value
        opt_E.append(opt_E_e)
    
    return prob.value, opt_E

def find_nearest_Q(E, NPA_level=3, show_distance=False):
    '''
    Function to find the nearest (in 2-norm) point in the quantum set (characterised by the NPA hierarchy with level 'NPA_level')
    from the point E, which is outside of the quantum set.
    '''
    E_flat = E.flatten()
    dim_tot = dimA*dimB*dimX*dimY

    ## VARIABLES
    s = cp.Variable()
    s_var_name = str(s.name())
    Evector = [cp.Variable() for i in range((dimX+1)*(dimY+1))]

    var_matching_rule = {}
    for i, var in enumerate(Evector):
        var_matching_rule[str(var.name())] = i

    Evector = np.reshape(Evector, (dimX+1,dimY+1))
    ## CONSTRAINTS
    constraints = []
    
    # Normalisation condition for pd
    for x,y in it.product(range(dimX+1),range(dimY+1)):
        if x==0 and y==0: continue
        constraints.append(Evector[x,y] <= 1)
        constraints.append(Evector[x,y] >= -1)

    # Positivity constraint
    for a,b,x,y in it.product(range(dimA),range(dimB),range(dimX),range(dimY)):
        p_abxy = (1+(-1)**a*Evector[x+1,0]+(-1)**b*Evector[0,y+1]+(-1)**(a+b)*Evector[x+1,y+1])/4
        constraints.append(p_abxy >= 0)

    # NPA constraint
    pd = [cp.Variable() for i in range(dim_tot)]
    pd = E2p_bi(Evector, np.reshape(pd, (dimA,dimB,dimX,dimY)), cvxpy=True)
    NPA_matrix = generate_NPA_matrix(pd, level=NPA_level)
    constraints.append(NPA_matrix >> 0)

    # 2-norm distance constraint - followed the program in [Appendix C, Physical Review A, 97(3):032309 (2018)].
    Matrix = []
    identityM = np.identity(len(E_flat))
    diff = E_flat-Evector.flatten()
    for i in range(len(E_flat)):
        row = [s*identityM[i][j] for j in range(len(E_flat))]
        row.append(diff[i])
        Matrix.append(row)
    last_row = list(diff)
    last_row.append(s)
    Matrix.append(last_row)

    Matrix = cp.bmat(Matrix)
    constraints.append(Matrix >> 0)

    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(verbose=False, solver='MOSEK')
    
    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')

    if show_distance: print("Distance between the original and the regularised data :", prob.value)
    if not s_var_name == str(prob.variables()[0].name()):
        raise Exception("The order of variables has been messed up. Please have a closer look.")

    opt_sol = [0]*len(E_flat)
    for variable in prob.variables()[1:]:
        if str(variable.name()) in var_matching_rule.keys():
            opt_sol[var_matching_rule[str(variable.name())]] = variable.value

    return np.reshape(opt_sol, (dimX+1,dimY+1))


def bi_max_Q_violation(Bell_ineq_in_E, print_result=False, NPA_level=3):
    '''
    Function to calculate the maximum quantum violation for a given Bell inequality 'Bell_inequality_in_E'.
    
    Inputs - Bell_inequality_in_E: the given Bell inequality in the correlator representation
           - print_result: whether to print out the details of the optimisation
           - NPA_level: the level of the NPA hierarchy to use
    Output - prob.value: the maximum quantum Bell violation
           - opt_Evector: the optimal quantum strategy to acheive the maximum violation 
    
    This function requires convex optimisation. We use the library 'cvxpy' with solver 'MOSEK'.
    '''
    # Fixed parameters
    dim_tot = dimA*dimB*dimX*dimY

    ## OPTIMISATION VARIABLES
    E = [cp.Variable() for i in range((dimX+1)*(dimY+1))]
    E = np.reshape(E, (dimX+1,dimY+1))

    ## OBJECTIVE FUNCTION
    object_func =  E.flatten()@Bell_ineq_in_E.flatten()

    ## OPTIMISATION CONSTRAINTS
    constraints = []
    constraints.append(E[0,0] == 1) # Constant part

    # Positivity constraint (non-signalling constraints)
    for a,b,x,y in it.product(range(dimA),range(dimB),range(dimX),range(dimY)):
        p_abxy = (1 + (-1)**a*E[x+1,0] + (-1)**b*E[0,y+1] + (-1)**(a+b)*E[x+1,y+1])/4
        constraints.append(p_abxy >= 0)

    # NPA constraint
    pd = [cp.Variable() for i in range(dim_tot)]
    pd = E2p_bi(E, np.reshape(pd, (dimA,dimB,dimX,dimY)), cvxpy=True)
    NPA_matrix = generate_NPA_matrix(pd, level=NPA_level)
    constraints.append(NPA_matrix >> 0)

    ## SOLVE OPTIMISATION
    prob = cp.Problem(cp.Maximize(object_func), constraints)
    prob.solve(verbose=print_result, solver='MOSEK')
    
    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')
    
    ## Extract the optimal solution
    opt_E = np.zeros((dimX+1,dimY+1))
    for x,y in it.product(range(dimX+1),range(dimY+1)):
        opt_E[x,y] = E[x,y].value
    
    return prob.value, opt_E

def PEF_opt_bi_2out_SQP(P_reg, P_in, Poly, Poly_in, beta, niter=3):
    '''
    PEF optimisation using the Scipy library.
    '''
    P_joint = np.zeros(np.shape(P_reg))
    for a,b,x,y in it.product(np.arange(2),repeat=4):
        P_joint[a,b,x,y] = P_reg[a,b,x,y]*P_in[x,y]
    P_joint = P_joint.flatten()
        
    def obj_func(F):
        # return -np.dot(np.log2(F), P_joint)
        return -sum([i*j for i,j in zip(np.log2(F), P_joint)])
    
    def gen_constraints(F):
        consts = []
        for E in Poly:
            p = E2p_bi(np.reshape(E,(3,3))).flatten()
            for i in range(len(p)):
                if p[i]<0: p[i]=0
            for p_in in Poly_in:
                LHS = sum([p[cz]**(beta+1)*F[cz]*p_in[(cz//2)%2, cz%2] for cz in range(16)])
                consts.append(1-LHS)
        return consts
    
    bounds = Bounds([1e-6]*16,[np.inf]*16)
    Constraints = ({'type': 'ineq', 'fun': lambda x: gen_constraints(x)})
    res_list = []
    x_list = []
    F0 = [1]*16
    for i in np.arange(niter):
        res = minimize(obj_func, F0, method='SLSQP',
                   constraints=Constraints, options={'ftol': 1e-10, 'disp': False},
                   bounds=bounds)
        if res.success:
            res_list.append(-obj_func(res.x))
            x_list.append(np.reshape(res.x, (dimA,dimB,dimX,dimY)))
        F0 = [rand.uniform(0.01,2) for _ in np.arange(16)]
    
    return max(res_list)/beta, x_list[np.argmax(res_list)]

def PEF_opt_bi_2out(P_reg, P_in, Poly, Poly_in, beta, show=False):
    '''
    PEF optimisation using 'cvxpy' with MOSEK.
    '''
    ## Variables
    F = [cp.Variable(nonneg=True) for _ in range(dimA*dimB*dimX*dimY)]
    F = np.reshape(F, (dimA,dimB,dimX,dimY))

    ## Objective function
    obj_func = sum([P_in[x,y]*P_reg[a,b,x,y]*cp.log(F[a,b,x,y])/np.log(2) for a,b,x,y 
                   in it.product(range(dimA),range(dimA),range(dimX),range(dimY))])

    ## Constraints
    constraints = []

    # The PEF constraint
    for E in Poly:
        p = E2p_bi(np.reshape(E,(3,3)))
        for a,b,x,y in it.product(range(dimA),range(dimA),range(dimX),range(dimY)):
            if p[a,b,x,y]<0: p[a,b,x,y] = 0
        for p_i in Poly_in:
            LHS = sum([p_i[x,y]*p[a,b,x,y]**(beta+1)*F[a,b,x,y] for a,b,x,y
                      in it.product(range(dimA),range(dimB),range(dimX),range(dimY))])
            constraints.append(LHS <= 1)

    ## Solve the problem
    prob = cp.Problem(cp.Maximize(obj_func), constraints)
    prob.solve(verbose=show, solver='MOSEK')

    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')

    ## Extract the optimal PEF
    optF = np.zeros((2,2,2,2))
    for a,b,x,y in it.product(range(dimA),range(dimB),range(dimX),range(dimY)):
        optF[a,b,x,y] = F[a,b,x,y].value

    return prob.value/beta, optF