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


shots = 10
q = 10
p = 100
d = q

timeflag = time.time()
P = random_Ham(p,q,d)
constants = [random.uniform(-1,1) for a in range(p)]
# print_Ham_string(P,constants)
print("Paulis:",p)
print("Qubits:",q)
print()
setup_time = time.time()-timeflag

timeflag = time.time()
m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(p))
# print(m.toarray())
pauli_to_matrix_time = time.time()-timeflag

timeflag = time.time()
psi = ground_state(m)
# print(psi)
ground_state_time = time.time()-timeflag

timeflag = time.time()
A = variance_graph(P,constants,psi)
# A.print()
variance_graph_time = time.time()-timeflag

timeflag = time.time()
aaa = greedy_min_parts(A)
# print(aaa)
clique_cover_time = time.time()-timeflag

diag_circuit_time = 0
diag_unitary_time = 0
diag_psi_time = 0
diag_prob_dist_time = 0
diag_sample_time = 0
for _ in range(shots):
    aa = aaa[random.randint(0,len(aaa)-1)]

    timeflag = time.time()
    P_aa = restrict_to_paulis(P,aa)
    C_aa = diagonalize(P_aa)
    Q_aa = P_aa.copy()
    act(Q_aa,C_aa)
    print(Q_aa.is_IZ())
    diag_circuit_time += time.time()-timeflag

    timeflag = time.time()
    U_aa = C_aa.unitary()
    diag_unitary_time += time.time()-timeflag

    timeflag = time.time()
    psi_aa = U_aa @ psi
    diag_psi_time += time.time()-timeflag

    timeflag = time.time()
    probs = [np.absolute(b)**2 for b in psi_aa]
    diag_prob_dist_time += time.time()-timeflag

    timeflag = time.time()
    sample = sample_from_distribution(probs,Q_aa.qubits())
    for b in range(Q_aa.paulis()):
        measurement = measurement_outcome(sample,Q_aa.a_pauli(b))
        pmdict = {1:"+1",-1:"-1"}
        print(pauli_to_string(P.a_pauli(aa[b])),":",pmdict[measurement])
    print()
    diag_sample_time += time.time()-timeflag

diag_circuit_time /= shots
diag_unitary_time /= shots
diag_psi_time /= shots
diag_prob_dist_time /= shots
diag_sample_time /= shots


print("Setup time:     ",setup_time)
print("Pauli to matrix:",pauli_to_matrix_time)
print("Ground state:   ",ground_state_time)
print("Variance graph: ",variance_graph_time)
print("Clique cover:   ",clique_cover_time)
print("Diag circuit:   ",diag_circuit_time)
print("Diag unitary:   ",diag_unitary_time)
print("Diag psi:       ",diag_psi_time)
print("Diag prob dist: ",diag_prob_dist_time)
print("Diag sample:    ",diag_sample_time)



