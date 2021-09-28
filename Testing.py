from Functions import *

# WORKING CODE

# Runtime notes:
	# Pauli Matrix time grows linearly with p, exponentially with q (q=18 max)
	# Ground State time grows exponentially with q (q=18 max)
	# Variance Graph time grows quadratically with p, exponentially with q (p=1000,q=18 max)
	# Commutation Graph time grows quadratically with p (p=5000 max)
	# Greedy Minimum Parts time grows linearly with p (p=20000 max)
	# Greedy Minimum Shots grows linearly with p (p=20000 max)
	# Anneal Minimum Shots grows quadratically with p (p=1000 max)



q = 10
p = 100
d = q

P = random_Ham(p,q,d)
constants = [random.uniform(-1,1) for a in range(p)]
# print_Ham_string(P,constants)
# print()
print("Paulis:",p)
print("Qubits:",q)
print()

m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(p))
# print(m.toarray())
# print()

psi = ground_state(m)
# print(psi)
# print()

A = variance_graph(P,constants,psi)
# A.print()
# print()

aaa = greedy_min_parts(A)
# print(aaa)
# print()

aa = aaa[random.randint(0,len(aaa)-1)]
Q = restrict_to_paulis(P,aa)
C = diagonalize(Q)
act(Q,C)
# Q.print()
# print()
# C.print()
# print()

cdf = distribution(C,psi)
# print(cdf)
# print()
# print([cdf[i+1]-cdf[i] for i in range(1<<q)])
# print()

sample = sample_from_distribution(cdf)
for b in range(Q.paulis()):
    measurement = measurement_outcome(sample,P.a_pauli(b),C)
    print(pauli_to_string(P.a_pauli(aa[b])),":",{1:"+1",-1:"-1"}[measurement])
print()










