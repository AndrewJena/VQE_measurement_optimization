from Functions import *

# WORKING CODE

# Runtime notes:
	# Pauli Matrix time grows linearly with p, exponentially with q (q=12 max)
	# Ground State time grows exponentially with q (q=12 max)
	# Variance Graph time grows quadratically with p, exponentially with q (p=1000,q=12 max)
	# Commutation Graph time grows quadratically with p (p=5000 max)
	# Greedy Minimum Parts time grows linearly with p (p=20000 max)
	# Greedy Minimum Shots grows linearly with p (p=20000 max)
	# Anneal Minimum Shots grows quadratically with p (p=1000 max)


error = None
shots = 100
p = 10
q = 4
d = q

P = very_random_Ham(p,q,d)
constants = [random.uniform(-1,1) for a in range(p)]
# constants = [1 for a in range(p)]
print_Ham_string(P,constants)
print("Paulis:",p)
print("Qubits:",q)
print()

m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(p))
# print(m)

psi = ground_state(m)
# print(psi)

A = variance_graph(P,constants,psi)
# print_graph(A)
# print()


# aaa = greedy_min_parts(A)
# # print(aaa)
# print("Greedy Minimum Parts Algorithm")
# print("Number of Parts:",len(aaa))
# print("Cost:",cost(A,aaa,error=error,shots=shots))
# print()

# aaa = greedy_min_shots(A,error=error,shots=shots)
# # print(aaa)
# print("Greedy Minimum Shots Algorithm")
# print("Number of Parts:",len(aaa))
# print("Runtime:",time.time()-start)
# print("Cost:",cost(A,aaa,error=error,shots=shots))
# print()

# aaa = anneal_min_shots(A,error=error,shots=shots)
# # print(aaa)
# print("Anneal Minimum Shots Algorithm")
# print("Number of Parts:",len(aaa))
# print("Runtime:",time.time()-start)
# print("Cost:",cost(A,aaa,error=error,shots=shots))
# print()

# aaa = greedy_edge_clique_cover(max_covariance_allowed(A,1))
# # print(aaa)
# print("Greedy Edge Clique Cover Algorithm")
# print("Number of Parts:",len(aaa))
# print("Cost:",cost(A,aaa,error=error,shots=shots))
# print()

# aaa = maximal_cliques(commutation_graph(P))
# # print(aaa)
# # print(aaa)
# print("Maximal Cliques Algorithm")
# print("Number of Parts:",len(aaa))
# # print("Cost:",cost(A,aaa,error=error,shots=shots))
# start = time.time()
# print("Cost0:",cost_allow_zero(A,aaa,error=error,shots=shots))
# print(time.time()-start)
# start = time.time()
# print("Costnew:",cost_new(A,aaa,error=error,shots=shots))
# print(time.time()-start)
# print()

# aaa = nonempty_cliques(commutation_graph(P))
# # print(aaa)
# print("Nonempty Cliques Algorithm")
# print("Number of Parts:",len(aaa))
# start = time.time()
# print("Cost1:",cost(A,aaa,error=error,shots=shots))
# print(time.time()-start)
# print()
# start = time.time()
# print("Cost2:",cost_old(A,aaa,error=error,shots=shots))
# print(time.time()-start)
# print()


# probs = np.zeros((p,p))
# variances = variance_graph(P,constants,psi).adj
# for a in range(p):
# 	for b in range(p):
# 		if a == b:
# 			probs[a,b] = (Mean(P.a_pauli(a),psi).real+1)/2
# 		else:
# 			q = variances[a,b]/4
# 			probs[a,b] = q

# print(probs)



# def sample_from_Ham(P):
# 	no

# def Mean_est(rxo,rx1):
# 	return (rx0-rx1)/(rx0+rx1+2)

# def Var_est(rx0,rx1):
# 	return 4*(rx0+1)/(rx0+rx1+2)*(rx1+1)/(rx0+rx1+2)

# def Cov_est(s00,s01,s10,s11,rx0,rx1,ry0,ry1):
# 	px = (rx0+1)/(rx0+rx1+2)
# 	py = (ry0+1)/(ry0+ry1+2)
# 	m0 = min(px*py,(1-px)*(1-py))
# 	m1 = min(px*(1-py),(1-px)*py)
# 	t00 = s00/((px)*(py))
# 	t01 = s01/((px)*(1-py))
# 	t10 = s10/((1-px)*(py))
# 	t11 = s11/((1-px)*(1-py))
# 	q = m0*m1*(t00-t01-t10+t11)/(m0*t00+m1*t01+m1*t10+m0*t11+2)
# 	return 4*q




# print(Cov_est(s00,s01,s10,s11,rx0,rx1,ry0,ry1))



error = None
shots = 50
p = 7
q = 4
d = q

old = 0
new = 0
for _ in range(50):
	P = very_random_Ham(p,q,d)
	constants = [random.uniform(-1,1) for a in range(p)]
	m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(p))
	psi = ground_state(m)
	A = variance_graph(P,constants,psi)
	aaa = nonempty_cliques(commutation_graph(P))
	start = time.time()
	cost(A,aaa,error=error,shots=shots)
	old += time.time()-start
	start = time.time()
	cost_new(A,aaa,error=error,shots=shots)
	new += time.time()-start
print(old)
print(new)

