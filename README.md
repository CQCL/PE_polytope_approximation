# PE_polytope_approximation

This project presents example codes for finite-data analysis for device-independent (DI) randomness verification using probability estimation technique [1] and using the algorithms for constructing polytope-approximations to the quantum set developed in [2]. Please see [2] for more details of how the analysis works.

This project is created by Quantinuum's Quantum Cryptography team. The codes are available for non-commercial use only; see the license for details. 

The codes require implementation of convex optimisation with semidefinite constraints, for which we use 'cvxpy' library with default solver 'MOSEK' (only free for academic purposes), and vertex & facet enumeration algorithms, for which we use 'pycddlib' library.

The project includes codes for two specific DI setups: 1) global randomness in CHSH setting (i.e., two parties with two dichotomic measurements on each party), and 2) two-output randomness in Mermin setting (i.e., three parties with two dichotomic measurements on each party).

The primary object of our codes is a class called 'PE_analysis' in the file 'tri-or-bi_PE_polytope.py'. An instance of this class can be created with arguments (typical correlation, input distribution, the choice of NPA hierarchy level, input distribution polytope). 'Typical correlation' represents the typical behaviour of the devices which can be obtained in the characterisation stage before actual runs of the protocol - for which the entropy witness will be obtained/optimised. 'Input distribution' describes the probability distribution of the input entropy source. 'The choice of NPA hierarchy level' is the NPA level that one wants to use in the anlysis. 'Input distribution polytope' describes the set (should be a polytope) of allowed input distributions if the input distribution is not fixed ('None' if the input distribution is fixed and independent).

The class 'PE_analysis' can:   
(a) Construct outer polytope approximations to the quantum set. One can use 'NearV' and 'MaxGP' algorithms developed in [2] to construct taylored approximations for the typical correlation. If there is any already known polytope approximation, one can add it to the class.   
(b) Find optimal probability estimation factors (PEFs) for the typical correlation with fixed PEF powers using polytope-approximations generated in (a). This process corresponds to the simpler optimisation in Eq.(18) in [2].   
(c) For some given data, calculate the extractable entropy using the optimised PEFs in (b).

Please see 'CHSH_example.ipynb', 'Asymmetric_CHSH.ipynb', and 'randomness_amplification.ipynb' for more specific example codes showing how to use the class 'PE_analysis'.

The Mermin case typically requires a large memory for PEF optimisation in (b) due to the large amount of constraints coming from many vertices of complex, high-dimensional polytope-approximations in the tripartite setting. For example, when we used a polytope constructed by one iteration of 'NearV' or 'MaxGP' algorithm, each PEF optimisation required ~30 GB memory and took 3-5 hours. 

Note that the codes for 'MaxGP' algorithm in the Mermin setting only adds one quantum Bell inequality per each iteration, in comparison to the CHSH setting where we add all inequalities generated from optimal strategies outside of the quantum set. This is because adding one inequality (already) massively increases the complexity of the resulting polytope. The 'MaxGP' algorithm in the Mermin setting looks at all optimal strategies (only the ones outside of the quantum set), and picks the best performing one in PEF optimisation.

When finding optimal PEFs with fixed powers in (b) in the CHSH setting, one can choose two different optimisation methods: 1) using 'SLSQP' method in SciPy library and 2) using 'cvxpy' library with 'MOSEK' solver. 1) is much slower than 2) and sometimes obtains invalid optimal PEFs due to numerical imprecision (which need some adjustment - see the discussion after Eq.(162) in [1]), but can deal with very small PEF power (near zero). Our codes normally use 2) for its numerical precision and the speed, but one can freely choose the option 1).

The NPA relaxations are implemented in the file 'NPA_lib.py' written by Hyejung H. Jee.

[1] E. Knill et al., Phys. Rev. R., 2(3):033465, (2020), Y. Zhang et al., Phys. Rev. A, 98(4):040304, (2018).   
[2] Our paper - to be published.
