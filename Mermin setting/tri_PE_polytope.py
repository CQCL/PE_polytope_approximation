import numpy as np
import cdd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from tri_lib import *

class PE_analysis:
	'''
	Class performing security analysis using Probability Estimation.
	Defined by some expected probability distributions 'P_reg' and 'P_in' (possibly) obtained from a characterisation stage.
	Generate some polytope approximations for the quantum set, and find optimal PEFs for such polytopes and 'P_reg'.
	Then, in function 'randomness_accumulation', calculate the extractable entropy rate for given data 'N_raw' using the optimised PEFs.
	'''
	def __init__(self, P_reg, P_in, NPA_level=2, Poly_in=None):
		self.P_reg = P_reg # Expected P from characterisation stage (should be in Q, i.e., regularised) - conditional form: P_reg[a,b,c,x,y,z] = P(abc|xyz)
		self.P_in = P_in # Input distribution: P_in[x,y,z] = p_in(x,y,z)
		self.E_reg = p2E_tri(self.P_reg) # P_reg in the correlator form
		self.rel_Mermin_ineq = [] # The relevant CHSH inequality
		self.q_rel_Mermin_ineq = [] # The relevant quantum CHSH inequality (with Tsirelson's bound)
		self.find_rel_Mermin_ineq()
		self.NPA_level = NPA_level # NPA level to use
		self.Poly = {
			"NS": [],
			"NearV": [],
			"MaxGP": []
		}
		self.logPR_list = {
			"NS": [],
			"NearV": [],
			"MaxGP": []
		} # Determined by P_reg
		self.power_list = []
		self.max_power = 0.1
		self.optF_power_list = {
			"NS": [],
			"NearV": [],
			"MaxGP": []
		} # Determined by P_reg
		self.Data_list = dict() # Collect and save any randomness accumulation result with data
		if Poly_in==None: self.Poly_in = [P_in] # Input polytope in case of randomness amplification
		else: self.Poly_in = Poly_in 

	def find_rel_Mermin_ineq(self):
		'''
		Find a relevant Mermin inequality which P_reg violates.
		'''
		for ineq in Mermin_ineqs:
			value = np.einsum('ijk,ijk->', ineq, self.E_reg)
			if value<0:
				self.rel_Mermin_ineq = ineq.copy()
				break
		self.q_rel_Mermin_ineq = self.rel_Mermin_ineq.copy()
		self.q_rel_Mermin_ineq[0][0] = 4

	def add_Poly_option(self, option, Poly):
		'''
		Function to add a new polytope option apart from {'NS', 'NearV', 'MaxGP'}
		'''
		self.Poly[option] = Poly
		self.optF_power_list[option] = [[]]*len(Poly)
		self.logPR_list[option] = [[]]*len(Poly)

	def show_Poly_options(self):
		'''
		Show all polytope options which have been already generated.
		'''
		avail_options = []
		for option in self.Poly.keys():
			if not self.Poly[option] == []:
				avail_options += [option+'('+str(len(self.Poly[option]))+')']
		print(avail_options)

	def generate_Poly(self, option, niter=1, cutoff=1, prob=None, fix_input=False):
		'''
		A function generating the polytope of option='option'.
		INPUT - option: the option name
				niter: the number of iterations (for 'NearV' and 'MaxGP')
				cutoff: the number of vertices to be considered in random choice (for 'NearV')
				prob: the probability distributions to choose a vertex in 'NearV'
				fix_input: when solving P_guess for some fixed input in 'MaxGP'
		'''
		self.optF_power_list[option] = [[]]*niter
		self.logPR_list[option] = [[]]*niter
		if option == "NS":
			self.Poly[option] = [NS_V_rep.copy()]
		elif option == "NearV":
			self.Poly[option] = self.generate_Poly_with_NearV(niter, cutoff, prob=prob)
		elif option == "MaxGP":
			self.Poly[option] = self.generate_Poly_with_MaxGP(niter, fix_input=fix_input) 

	def generate_Poly_with_NearV(self, niter=1, cutoff=1, prob=None):
		'''
		Generate poltyopes following 'NearV' algorithm.
		'''
		print("=================== NearV algorithm starts ===================")
		if self.Poly["NS"] == []:
			print("Please generate 'NS' polytope first.")
			return

		GivenH = positivity_NS.copy()
		Poly_i = self.Poly["NS"][0].copy()
		Poly_list = []
		for i in range(niter):
			print("Iteration {} starts".format(i))

			norm_list = []
			for ext in Poly_i:
				p = E2p_tri(np.reshape(ext, (3,3,3)))
				norm_list.append(norm(p.flatten(), self.P_reg.flatten()))

			m = min(cutoff, len(norm_list))

			ind_nearest_b = np.argsort(norm_list)
			ind_nearest = []
			j = 0
			while len(ind_nearest)<m:
				if j>=len(ind_nearest_b):
					raise Exception("There are not enough extreme points for the cutoff.")
					return

				p = E2p_tri(np.reshape(Poly_i[ind_nearest_b[j]], (3,3,3)))
				try: is_quantum = check_quantum_NPA(p, level=2)
				except:
					print("MOSEK failed -  too close to the quantum set.")
					is_quantum = True
				if not is_quantum:
					ind_nearest.append(ind_nearest_b[j])
				j += 1

			if m>1: 
				if prob == None:
					normC = 0
					for i in range(m):
						normC += 1/norm_list[ind_nearest[i]]
					prob = []
					for i in range(m):
						prob = prob + [1/norm_list[ind_nearest[i]]/normC]

				ext_ind = rand.choice(list(range(m)), 1, p=prob[:m])[0]
			else:
				ext_ind = 0

			print("Chosen index:", ext_ind)
			ext = np.reshape(Poly_i[ind_nearest[ext_ind]], (3,3,3))

			new_ineq = self.generate_q_BI_from_E(ext)
			Poly_i = self.generate_V_from_GivenH(GivenH+[new_ineq])

			GivenH.append(new_ineq)
			Poly_list.append(Poly_i)
			print("---------------------")

		return Poly_list

	def generate_Poly_with_MaxGP(self, niter=1, fix_input=False):
		'''
		Generate polytopes following 'MaxGP' algorithm.
		'''
		print("=================== MaxGP algorithm starts ===================")
		GivenH = positivity_NS.copy()
		Poly_i = []
		Poly_list = []

		for i in range(niter):
			if fix_input == False:
				x,y = [random.randint(0,1) for _ in range(2)]
			else:
				x,y = fix_input
			GP, GP_optE = GP_tri_2out_fullS_GivenH(self.E_reg, GivenH, xbar=x, ybar=y)

			new_ineq_list_i = []
			Poly_list_i = []
			max_count = 0
			max_logPR = 0
			skip_count = 0

			print("Iteration {} starts".format(i))
			print("Chosen input for guessing probability:", x,y)
			# In tripartite setting, we only add one quantum Bell inequality per iteration
			# because adding one linear inequality massively increases the complexity.
			# We try all optimal strategies and pick the best performing one in PEF optimisation.
			for j in range(len(GP_optE)):
				print("Generating the polytope for optimal solution no.{}...".format(j+1))
				is_quantum = check_quantum_NPA(E2p_tri(GP_optE[j]), level=3)
				if is_quantum: 
					print("Optimal solution no.{} is quantum. Continue to the next one.".format(j+1))
					skip_count += 1
					continue

				new_ineq = self.generate_q_BI_from_E(GP_optE[j])
				new_ineq_list_i.append(new_ineq)

				Poly_with_new_ineq = self.generate_V_from_GivenH(GivenH+[new_ineq])
				Poly_list_i.append(Poly_with_new_ineq)

				power = 0.01
				print("Optimising PEF...")
				logPR, optF = PEF_opt_tri_2out(self.P_reg, self.P_in, Poly_with_new_ineq, self.Poly_in, power)
				if max_logPR<logPR:
					max_logPR = logPR
					max_count = j - skip_count

			GivenH.append(new_ineq_list_i[max_count])
			Poly_i = Poly_list_i[max_count]
			Poly_list.append(Poly_i)
			print("---------------------")

		return Poly_list

	def get_optF(self, option, n_list=None, first_power=0.00015, n_points=21, power_list=None):
		'''
		Optimise PEF for fixed power in 'power_list'.
		The outcome is a list of optimised PEFs - each PEF is optimal for each power in 'power_list'.
		Optimised for 'P_reg' (possibly) obtained from the characterisation stage.
		Input - option: polytope option such as 'NearV' or 'NS'
				n_list: a list of all numbers of iterations for 'option' for which case we want to optimse PEFs.
				power_list: the list of powers for which we optimse PEFs. When 'None', the function creates a list
							from 'first_power' and 'n_points'.
				first_power: the first power in the created power_list, if 'power_list' is not given.
				n_points: the number of points in the created power_list, if 'power_list' is not given.
				MOSEK: if 'True', we use cvxpy with MOSEK for PEF optimisation (quicker than other option).
					   if 'False', we use SciPy library for PEF optimisation (more accurate if we need more precision near power=0, but slow).
		'''
		print("=================== Optimise PEF for '{}' ===================".format(option))
		if n_list == None:
			n_list = list(range(len(self.Poly[option])))
		for n in n_list:
			if self.Poly[option][n] == []:
				print("The polytope for option '{}' with '{}' iteration(s) has not been generated yet.".format(option, n))
				return
		if power_list == None:
			self.power_list = np.linspace(0,self.max_power,n_points)
			self.power_list[0] = first_power
		else:
			self.power_list = power_list

		for n in n_list:
			polytope = self.Poly[option][n]

			optF_power_list = []
			logPR_list = []
			for power in tqdm(self.power_list):
				logPR, optF = PEF_opt_tri_2out(self.P_reg, self.P_in, polytope, self.Poly_in, power)

				optF_power_list.append(optF)
				logPR_list.append(logPR)
			self.optF_power_list[option][n] = optF_power_list
			self.logPR_list[option][n] = logPR_list

	def generate_q_BI_from_E(self, E):
		'''
		Generate a (tangent) quantum Bell inequality from given point 'E' outside of the quantum set.
		For given 'E', we find the closest point in the quantum set to E
					   and define the Bell coefficients by the difference between the two.
		'''
		nearest_Q = find_nearest_Q(E)
		bell_coeff = np.round(E - nearest_Q, decimals=6)
		bell_coeff = bell_coeff*10
		max_Q_value, max_Q_point = tri_max_Q_violation(bell_coeff)

		new_ineq = -bell_coeff.copy()
		new_ineq[0,0,0] = np.round(max_Q_value, decimals=6)+10**(-6)
		return new_ineq

	def generate_V_from_GivenH(self, GivenH):
		'''
		Generate V-representation of the polytope from its H-representation ('GivenH').
		'''
		H_rep = []
		for ineq in GivenH:
			H_rep.append([str(i) for i in ineq.flatten()])

		mat = cdd.Matrix(H_rep, number_type='fraction')
		mat.rep_type = cdd.RepType.INEQUALITY
		poly = cdd.Polyhedron(mat)
		V_rep_fraction = np.array(poly.get_generators())

		V_rep = []
		for ext in V_rep_fraction:
			V_rep.append(np.array([float(i) for i in ext]))
		return V_rep

	def plot_logPR_vs_power(self):
		'''
		Plot logPR vs power for all option(s) for which PEFs are already optimised.
		'''
		legends = []
		for option in self.logPR_list.keys():
			if not self.logPR_list[option] == []:
				for i in np.arange(len(self.logPR_list[option])):
					plt.plot(self.power_list, self.logPR_list[option][i])
					if len(self.logPR_list[option])==1: legends.append(option)
					else: legends.append(option+' '+str(i+1)+' iterations')
		plt.xlim([0,self.max_power])
		plt.legend(legends)
		plt.show()

		return opt_netEntR, opt_beta

	def randomness_accumulation(self, N_raw, eps_sec, option, n_list=None, show_result=False, save=False, title=None):
		'''
		Calculate the (accumulated) extractable entropy for given data 'N_raw'.
		The function involves
		a. Finding the optimal PEF and power for given N (which can be calculated from 'N_raw') and 'eps_sec'.
		b. Calculating the extractable entropy rate using the optimal PEF and power.

		Input - N_raw: given data. N[a,b,x,y] represents the counts for (a,b,x,y). The sum of all entries equals to N.
				eps_sec: the security parameter.
				option: polytope option such as 'MaxGP' or 'NS+CHSH'.
				n_list: a list of all numbers of iterations for 'option' that we want to use for accumulation.
				show_result: if True, print out results.
				save: if True, save the result in 'self.Data_list'.
				title: the 'key' of the saved result in 'self.Data_list'. If None, title = option.
		'''
		if n_list == None:
			n_list = list(range(len(self.optF_power_list[option])))
		for n in n_list:
			if self.optF_power_list[option][n] == []:
				print("Please optimise PEFs for option '{}' with '{}' iterations first.".format(option, n))
				return

		N = np.sum(N_raw)
		Data = []
		for n in n_list:
			extEntR_list = []
			optF_list = []
			for i, power in enumerate(self.power_list):
				optF = self.optF_power_list[option][n][i]

				optF_list.append(optF)
				extEntR_list.append(self.calculate_extEntR(N_raw, optF, power, N, eps_sec))
			# Optimise power
			i_opt = np.argmax(extEntR_list)
			opt_power = self.power_list[i_opt]
			optF = optF_list[i_opt]
			opt_extEntR = extEntR_list[i_opt]
			if show_result:
				print("============================ Randomness Accumulation using PE ============================")
				print("Using the polytope for option '{}' after {} iteration(s).".format(option, n+1))
				print("Optimal power:", opt_power)
				print("Optimal PEF:", optF)
				print("Accumulated entropy:", N*opt_extEntR)
				print("Final extractable entropy rate:", opt_extEntR)
				print("------------------------------------------------------------------------------------")

			small_data = {
			'option': option,
			'number of iterations': n+1,
			'entropy rate': opt_extEntR,
			'optimal power': opt_power,
			'optimal PEF': optF 
			}
			Data.append(small_data)

		if title==None:
			title = option
		if save:
			self.Data_list[title] = (option, N_raw, eps_sec, Data)
		return Data

	def calculate_extEntR(self, N_raw, optF, power, N, eps):
		'''
		Calculate the extractable entropy rate from the data 'N_raw', optimal PEF 'optF', optimal power 'power', total 'N', and the security parameter 'eps'.
		'''
		return np.einsum('ijklmn,ijklmn->', N_raw, np.log2(optF))/power/N + np.log2(eps/(1+power))/power/N + np.log2(power*eps/(1+power))/N
