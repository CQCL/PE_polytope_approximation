import numpy as np
import cvxpy as cp
import itertools as it

from NPA_lib import generate_NPA_matrix, check_quantum_NPA

### Mermin setting ###
dimA = 2
dimB = 2
dimC = 2
dimX = 2
dimY = 2
dimZ = 2

### All NS inequalities ###
positivity_NS = []
for a,b,c,x,y,z in it.product(*[range(2)]*6):
    pos_ineq = np.zeros((3,3,3))
    pos_ineq[0,0,0] = 1
    pos_ineq[x+1,0,0] = (-1)**a
    pos_ineq[0,y+1,0] = (-1)**b
    pos_ineq[0,0,z+1] = (-1)**c
    pos_ineq[x+1,y+1,0] = (-1)**(a+b)
    pos_ineq[x+1,0,z+1] = (-1)**(a+c)
    pos_ineq[0,y+1,z+1] = (-1)**(b+c)
    pos_ineq[x+1,y+1,z+1] = (-1)**(a+b+c)
    positivity_NS.append(pos_ineq)

#######################################
### All Mermin inequalities ###
with open('tri_Mermin_ineqs.npy', 'rb') as f:
    Mermin_ineqs = np.load(f, allow_pickle=True)

###############################################
### All NS_extreme points in Pironio et al. ###
with open('tri_NS_ext_points.npy', 'rb') as f:
    NS_V_rep_in_classes = np.load(f, allow_pickle=True)
    
NS_V_rep = []
for list_ext in NS_V_rep_in_classes:
    for ext in list_ext:
        NS_V_rep.append(ext.flatten())
###############################################

def tri_regularise_to_Q(P_raw, P_in, NPA_level=3, show_distance=False):
    '''
    A function to regularise the observed statistics P to the p which maximises the likelihood function.
    The method is discussed in [Physical Review A, 97(3):032309 (2018)].
    
    Inputs - P_raw: the observed statistics, numpy array P_raw[a,b,c,x,y,z] = P(a,b,c|x,y,z)
           - P_in: the observed input distribution, P_in[x,y,z] = p(x,y,z)
           - NPA_level: the level of the NPA hierarchy to use
           - show_distance: whether to print out the optimal distance between P and the closest p
    Output - p: the closest probability distribution in the quantum set
                numpy array p[a,b,c,x,y,z] = p(a,b,c|x,y,z)
    
    This function requires convex optimisation.
    We are using the library 'cvxpy' and the solver 'MOSEK'
    '''

    ## FIXED PARAMETERS
    dim_tot = dimA*dimB*dimC*dimX*dimY*dimZ
    dim_range_tot = (range(dimA),range(dimB),range(dimC),range(dimX),range(dimY),range(dimZ))

    ## OPTIMISATION VARIABLES
    pd = [cp.Variable(nonneg=True) for i in range(dim_tot)]
    pd = np.reshape(pd, (dimA,dimB,dimC,dimX,dimY,dimZ))

    ## OBJECTIVE FUNCTION for x='xbar' and y='ybar' 
    object_func =  sum([-P_in[x,y,z]*P_raw[a,b,c,x,y,z]*cp.log(pd[a,b,c,x,y,z])/np.log(2) for a,b,c,x,y,z in it.product(*dim_range_tot)])

    ## OPTIMISATION CONSTRAINTS
    constraints = []

    # Normalisation constraints
    pd_xyz = np.sum(pd, axis=(0,1,2))
    for x,y,z in it.product(range(dimX),range(dimY),range(dimZ)):
        constraints.append(pd_xyz[x,y,z] == 1)

    # No-signalling
    for b,c,y,z in it.product(range(dimB),range(dimC),range(dimY),range(dimZ)):
        no_sig_A2BC = []
        for x in range(dimX):
            marg_prog = sum([pd[a,b,c,x,y,z] for a in range(dimA)])
            no_sig_A2BC.append(marg_prog)
        for x in range(1,dimX):
            constraints.append(no_sig_A2BC[0] == no_sig_A2BC[x])
    for a,c,x,z in it.product(range(dimA),range(dimC),range(dimX),range(dimZ)):
        no_sig_B2AC = []
        for y in range(dimY):
            marg_prog = sum([pd[a,b,c,x,y,z] for b in range(dimB)])
            no_sig_B2AC.append(marg_prog)
        for y in range(1,dimY):
            constraints.append(no_sig_B2AC[0] == no_sig_B2AC[y])
    for a,b,x,y in it.product(range(dimA),range(dimB),range(dimX),range(dimY)):
        no_sig_C2AB = []
        for z in range(dimZ):
            marg_prog = sum([pd[a,b,c,x,y,z] for c in range(dimC)])
            no_sig_C2AB.append(marg_prog)
        for z in range(1,dimZ):
            constraints.append(no_sig_C2AB[0] == no_sig_C2AB[z])

    # NPA constraint
    NPA_matrix = generate_NPA_matrix(pd, level=NPA_level)
    constraints.append(NPA_matrix >> 0)

    ## SOLVE OPTIMISATION
    prob = cp.Problem(cp.Minimize(object_func), constraints)
    prob.solve(verbose=False, solver='MOSEK')
    
    P_reg_ML = np.zeros((2,2,2,2,2,2))
    for a,b,c,x,y,z in it.product(*dim_range_tot):
        P_reg_ML[a,b,c,x,y,z] = pd[a,b,c,x,y,z].value

    if not prob.status == cp.OPTIMAL: raise Exception('Not optimal.')
    if show_distance: 
        KL_div = prob.value + sum([P_in[x,y,z]*P_raw[a,b,c,x,y,z]*np.log2(P_raw[a,b,c,x,y,z]) for a,b,c,x,y,z in it.product(*dim_range_tot)])
        print("Final relative entropy :", KL_div)

    return P_reg_ML

def p2E_tri(p):
    '''
    Function to convert the conditional probability 'p' to its correlator representation E.
    
    Input - p: tripartite conditional probability, numpy array p[a,b,c,x,y,z] = p(a,b,c|x,y,z)
    Output - Evector: correlators, numpy array E[x,y,z] = <A_(x+1)B_(y+1)C_(z+1)> where A_0,B_0,C_0 are the identity.
    '''
    Evector = np.zeros((3,3,3))
    
    dims = np.shape(p)
    dimA = dims[:3]
    dimQ = dims[3:]
    dimE = [dim+1 for dim in dimQ]
    
    subindices = [range(dim+1) for dim in dimQ]
    for x,y,z in it.product(*subindices):
        if x==0 and y==0 and z==0: 
            Evector[x,y,z] = np.sum(p, axis=(0,1,2))[0,0,0]
            continue
            
        q_indices = [x,y,z]
        not_in = []
        yes_in = []
        for i in range(3):
            if q_indices[i]==0: not_in.append(i)
            else: yes_in.append(i)
                
        if len(yes_in)==1:
            sum_axis = tuple(not_in+[x+3 for x in not_in])
            p_summed = np.sum(p, axis=sum_axis)/4
            Evector[x,y,z] = sum([(-1)**a*p_summed[a,q_indices[yes_in[0]]-1] for a in range(2)])
        elif len(yes_in)==2:
            sum_axis = tuple(not_in+[x+3 for x in not_in])
            p_summed = np.sum(p, axis=sum_axis)/2
            Evector[x,y,z] = sum([(-1)**(a+b)*p_summed[a,b,q_indices[yes_in[0]]-1,q_indices[yes_in[1]]-1] 
                                  for a,b in it.product(range(2),repeat=2)])
        elif len(yes_in)==3:
            Evector[x,y,z] = sum([(-1)**(a+b+c)*p[a,b,c,x-1,y-1,z-1] for a,b,c, in it.product(range(2),repeat=3)])
        else:
            raise Exception("There occured some problem converting the probability distribution to correlators.")
    return Evector

def E2p_tri(Evector, p=0, cvxpy=False):
    '''
    Function to convert the correlators 'Evector' to its probability representation.
    
    Input - Evector: correlators, numpy array E[x,y,z] = <A_(x+1)B_(y+1)C_(z+1)>
    Output - p: conditional probability, numpy array p[a,b,c,x,y,z] = p(a,b,c|x,y,z)
    '''
    if not cvxpy:
        p = np.zeros((2,2,2,2,2,2))
    dimQ = [dim-1 for dim in np.shape(Evector)]
    subindices = [range(dim) for dim in dimQ]
    for a,b,c in it.product(*[range(2)]*3):
        for x,y,z in it.product(*subindices):
            p[a,b,c,x,y,z] = (1+(-1)**a*Evector[x+1,0,0]+(-1)**b*Evector[0,y+1,0]+(-1)**c*Evector[0,0,z+1]
                             +(-1)**(a+b)*Evector[x+1,y+1,0]+(-1)**(a+c)*Evector[x+1,0,z+1]+(-1)**(b+c)*Evector[0,y+1,z+1]
                             +(-1)**(a+b+c)*Evector[x+1,y+1,z+1])/8

            if cvxpy: continue
            if p[a,b,c,x,y,z]<0:
                if np.isclose(p[a,b,c,x,y,z],0):
                    p[a,b,c,x,y,z] = 0
                else:
                    raise Exception("Some probabilities are negative.")
    return p

def norm(P1, P2):
    if not len(P1)==len(P2):
        print("The two vectors must have the same length.")
    else:
        value = sum([abs(P1[i]-P2[i]) for i in range(len(P1))])/2 # Total variation distance
        # value = np.sqrt(sum([(P1[i]-P2[i])**2 for i in range(len(P1))])) # 2-norm
        return value

def GP_tri_2out_fullS_GivenH(E, given_ineqs, xbar, ybar, show=False):
    '''
    Solving optimisation calculating the guessing probability of 2 outputs for given inputs 'xbar, ybar'
    AGAINST adversaries characterised by 'given ineqs' in the tripartite setting using the full statistics.
    
    Inputs - E_data: the observed statistics in the correlator representation
                     numpy array E[x,y,z] = <A_(x+1)B_(y+1)C_(z+1)> with A_0=B_0=C_0=I
           - xbar: the fixed 'x'
           - ybar: the fixed 'y'
           - show: whether to print out the details of the optimisation
    Output - prob.value: the optimal guessing probability
           - opt_Evectors: the optimal solutions (optimal strategies for the adversary)
           
    This function requires convex optimisation. We use the library 'cvxpy' with solver 'MOSEK'
    '''
    ## Fixed parameters
    dim_E = (dimX+1)*(dimY+1)*(dimZ+1)

    ## Optimisation Variables
    # 1. correlators
    Evectors = []
    for e1,e2 in it.product(range(dimA),range(dimB)):
        E_e = [cp.Variable() for i in range(dim_E)]
        E_e = np.reshape(E_e, (dimX+1,dimY+1,dimZ+1))
        Evectors.append(E_e)
    Evectors = np.reshape(Evectors, (dimA,dimB,dimX+1,dimY+1,dimZ+1))

    # 2. normalisations
    norm = [cp.Variable(nonneg=True) for e in range(dimA*dimB)] # norm of each p_e
    norm = np.reshape(norm, (dimA,dimB))

    ## Objective Function for x=xbar, y=ybar
    elements = []
    for a,b in it.product(range(dimA),range(dimB)):
        pd = (Evectors[a][b][0,0,0] + (-1)**a*Evectors[a][b][xbar+1,0,0] + (-1)**b*Evectors[a][b][0,ybar+1,0] + (-1)**(a+b)*Evectors[a][b][xbar+1,ybar+1,0])/4
        elements.append(pd)
    object_func = sum(elements)

    ## Optimisation Constraints
    constraints = []

    # Observed statistics constraints
    avg_E = np.sum(Evectors, axis=(0,1))
    for x,y,z in it.product(range(dimX+1),range(dimY+1),range(dimZ+1)):
        constraints.append(avg_E[x,y,z] == E[x,y,z])

    # Normalisation conditions
    for e1,e2 in it.product(range(dimA),range(dimB)):
        constraints.append(norm[e1,e2] <= 1)
        for x,y,z in it.product(range(dimX+1),range(dimY+1),range(dimZ+1)):
            if x==0 and y==0 and z==0:
                constraints.append(Evectors[e1,e2][0,0,0] == norm[e1,e2]) # Constant part
            constraints.append(Evectors[e1,e2][x,y,z] <= norm[e1,e2])
            constraints.append(Evectors[e1,e2][x,y,z] >= -norm[e1,e2])
    constraints.append(np.sum(norm) == 1) # They should sum up to 1.

    for e1,e2 in it.product(range(dimA),range(dimB)):
        for i in range(len(given_ineqs)):
            value = given_ineqs[i].flatten()@Evectors[e1][e2].flatten()
            constraints.append(value >= 0)

    ## Solve the Optimisation
    prob = cp.Problem(cp.Maximize(object_func), constraints)
    prob.solve(verbose=show, solver='MOSEK')
    
    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')
    
    ## Extract the optimal solution
    opt_Evectors = []
    for e1,e2 in it.product(range(dimA),range(dimB)):
        opt_E = np.zeros((dimX+1, dimY+1, dimZ+1))
        for x,y,z in it.product(range(dimX+1),range(dimY+1),range(dimZ+1)):
            opt_E[x,y,z] = Evectors[e1][e2][x,y,z].value/norm[e1,e2].value
        opt_Evectors.append(opt_E)
    
    return prob.value, opt_Evectors

def find_nearest_Q(E, NPA_level=3, show_distance=False):
    '''
    Function to find the nearest (in 2-norm) point in the quantum set (characterised by the NPA hierarchy with level 'NPA_level')
    from the point E, which is outside of the quantum set.
    '''
    E_flat = E.flatten()

    dim_tot = dimA*dimB*dimC*dimX*dimY*dimZ

    ## OPTIMISATION VARIABLES
    s = cp.Variable()
    s_var_name = str(s.name())
    Evector = [cp.Variable() for i in range((dimX+1)*(dimY+1)*(dimZ+1))]

    var_matching_rule = {}
    for i, var in enumerate(Evector):
        var_matching_rule[str(var.name())] = i

    Evector = np.reshape(Evector, (dimX+1,dimY+1,dimZ+1))
    ## OPTIMISATION CONSTRAINTS
    constraints = []
    
    # Normalisation condition for pd
    for x,y,z in it.product(range(dimX+1),range(dimY+1),range(dimZ+1)):
        if x==0 and y==0 and z==0: continue
        constraints.append(Evector[x,y,z] <= 1)
        constraints.append(Evector[x,y,z] >= -1)

    # Positivity constraint
    for a,b,c,x,y,z in it.product(range(dimA),range(dimB),range(dimC),range(dimX),range(dimY),range(dimZ)):
        p_abcxyz = (1+(-1)**a*Evector[x+1,0,0]+(-1)**b*Evector[0,y+1,0]+(-1)**c*Evector[0,0,z+1]
                    +(-1)**(a+b)*Evector[x+1,y+1,0]+(-1)**(a+c)*Evector[x+1,0,z+1]+(-1)**(b+c)*Evector[0,y+1,z+1]
                    +(-1)**(a+b+c)*Evector[x+1,y+1,z+1])/8
        constraints.append(p_abcxyz >= 0)

    # NPA constraint
    pd = [cp.Variable() for i in range(dim_tot)]
    pd = E2p_tri(Evector, np.reshape(pd, (dimA,dimB,dimC,dimX,dimY,dimZ)), cvxpy=True)
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

    ## SOLVE OPTIMISATION
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(verbose=False, solver='MOSEK')
    
    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')

    if show_distance: print("Distance between the original and the regularised data :", prob.value)
    if not s_var_name == str(prob.variables()[0].name()):
        raise Exception("The order of variables has been messed up. Please have a closer look.")

    ## Extract the optimal solution
    opt_sol = [0]*len(E_flat)
    for variable in prob.variables()[1:]:
        if str(variable.name()) in var_matching_rule.keys():
            opt_sol[var_matching_rule[str(variable.name())]] = variable.value

    return np.reshape(opt_sol, (dimX+1,dimY+1,dimZ+1))

def tri_max_Q_violation(Bell_ineq_in_E, print_result=False, NPA_level=3):
    '''
    Function to calculate the maximum quantum violation for a given Bell inequality 'Bell_inequality_in_E'.
    
    Inputs - Bell_inequality_in_E: the given Bell inequality in the correlator representation
           - print_result: whether to print the details of the optimisation
           - NPA_level: the level of the NPA hierarchy to use
    Output - prob.value: the maximum quantum Bell violation
           - opt_Evector: the optimal quantum strategy to acheive the maximum violation 
    
    This function requires convex optimisation. We use the library 'cvxpy' with solver 'MOSEK'.
    '''
    # Fixed parameters
    dim_tot = dimA*dimB*dimC*dimX*dimY*dimZ

    ## OPTIMISATION VARIABLES
    Evector = [cp.Variable() for i in range((dimX+1)*(dimY+1)*(dimZ+1))]
    Evector = np.reshape(Evector, (dimX+1,dimY+1,dimZ+1))

    ## OBJECTIVE FUNCTION
    object_func =  Evector.flatten()@Bell_ineq_in_E.flatten()

    ## OPTIMISATION CONSTRAINTS
    constraints = []
    constraints.append(Evector[0,0,0] == 1) # Constant part

    # Positivity constraint (non-signalling constraints)
    for a,b,c,x,y,z in it.product(range(dimA),range(dimB),range(dimC),range(dimX),range(dimY),range(dimZ)):
        p_abcxyz = (1+(-1)**a*Evector[x+1,0,0]+(-1)**b*Evector[0,y+1,0]+(-1)**c*Evector[0,0,z+1]
                    +(-1)**(a+b)*Evector[x+1,y+1,0]+(-1)**(a+c)*Evector[x+1,0,z+1]+(-1)**(b+c)*Evector[0,y+1,z+1]
                    +(-1)**(a+b+c)*Evector[x+1,y+1,z+1])/8
        constraints.append(p_abcxyz >= 0)

    # NPA constraint
    pd = [cp.Variable() for i in range(dim_tot)]
    pd = E2p_tri(Evector, np.reshape(pd, (dimA,dimB,dimC,dimX,dimY,dimZ)), cvxpy=True)
    NPA_matrix = generate_NPA_matrix(pd, level=NPA_level)
    constraints.append(NPA_matrix >> 0)

    ## SOLVE OPTIMISATION
    prob = cp.Problem(cp.Maximize(object_func), constraints)
    prob.solve(verbose=print_result, solver='MOSEK')
    
    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')
    
    ## Extract the optimal solution
    opt_Evector = np.zeros((dimX+1,dimY+1,dimZ+1))
    for x,y,z in it.product(range(dimX+1),range(dimY+1),range(dimZ+1)):
        opt_Evector[x,y,z] = Evector[x,y,z].value
    
    return prob.value, opt_Evector

def PEF_opt_tri_2out(P_reg, P_in, Poly, Poly_in, beta, show=False):
    '''
    PEF optimisation using 'cvxpy' with MOSEK.
    '''
    dim_tot = dimA*dimB*dimC*dimX*dimY*dimZ
    tot_dim_list = (dimA,dimB,dimC,dimX,dimY,dimZ)
    tot_range_list = (range(dimA),range(dimB),range(dimC),range(dimX),range(dimY),range(dimZ))
    
    ## OPTIMISATION Variables
    F = [cp.Variable(nonneg=True) for i in range(dim_tot)]
    F = np.reshape(F, tot_dim_list)

    ## Objective function
    obj_func = sum([P_in[x,y,z]*P_reg[a,b,c,x,y,z]*cp.log(F[a,b,c,x,y,z]) for a,b,c,x,y,z in it.product(*tot_range_list)])

    ## OPTIMISATION Constraints
    constraints = []

    # The PEF constraint
    for E in Poly:
        p = E2p_tri(np.reshape(E, (dimX+1,dimY+1,dimZ+1)))
        for a,b,c,x,y,z in it.product(range(2),repeat=6):
            if p[a,b,c,x,y,z]<0: p[a,b,c,x,y,z] = 0
        p_abxyz = np.sum(p,axis=2)
        for p_i in Poly_in:
            LHS = sum([p_i[x,y,z]*p[a,b,c,x,y,z]*np.power(p_abxyz[a,b,x,y,0],beta)*F[a,b,c,x,y,z] 
                   for a,b,c,x,y,z in it.product(*tot_range_list)])
            constraints.append(LHS <= 1)

    ## Solve Optimisation
    prob = cp.Problem(cp.Maximize(obj_func), constraints)
    prob.solve(verbose=show, solver='MOSEK')

    if prob.status != cp.OPTIMAL: raise Exception('Not optimal.')
            
    ## Extract the optimal PEF
    opt_F = np.zeros(tot_dim_list)
    for a,b,c,x,y,z in it.product(*tot_range_list):
        opt_F[a,b,c,x,y,z] = F[a,b,c,x,y,z].value
    
    return prob.value/(np.log(2)*beta), opt_F 