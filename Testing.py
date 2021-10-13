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

shots = 1000


# 4 Qubits
# name = "4qubits_H2"

# 8 Qubits
name = "8qubits_H2"

# 12 Qubits
# name = "12qubits_LiH"

# 14 Qubits
# name = "14qubits_BeH2"
# name = "14qubits_H2O"

# 16 Qubits
# name = "16qubits_NH3"

# 20 Qubits
# name = "20qubits_C2"
# name = "20qubits_HCl"


form = "bk"
# form = "jw"
# form = "parity"


file_Hamiltonian = "./Hamiltonians/"+name+"/"+form+".txt"
file_ground_state = "./Hamiltonians/"+name+"/"+form+"_ground_state.txt"
file_variance_graph = "./Hamiltonians/"+name+"/"+form+"_variance_graph.txt"
file_maximal_cliques = "./Hamiltonians/"+name+"/"+form+"_maximal_cliques.txt"

p,q,P,constants = read_Hamiltonian(file_Hamiltonian)




# # print_Ham_string(P,constants)
# print()
# print("Paulis:",p)
# print("Qubits:",q)
# print()


# t_psi = time.time()
# if os.path.isfile(file_ground_state):
#     psi = read_ground_state(file_ground_state)
# else:
#     psi = ground_state(P,constants)
#     write_ground_state(file_ground_state,psi)
# # print(psi)
# # print()
# print("Psi Time:",time.time()-t_psi)


# t_A = time.time()
# if os.path.isfile(file_variance_graph):
#     A = read_variance_graph(file_variance_graph)
# else:
#     A = variance_graph(P,constants,psi)
#     write_variance_graph(file_variance_graph,A)
# # A.print()
# # A.print_neighbors()
# # print()
# print("A Time:  ",time.time()-t_A)


# t_aaa = time.time()
# # aaa = covering_maximal_cliques(A,1)
# aaa = maximal_cliques(A)
# # print(aaa)
# # print()
# print("aaa Time:",time.time()-t_aaa)


# t_X = time.time()
# X = greedy_bayes_min_var(P,constants,psi,aaa,shots)
# # print(X)
# # print()
# print("X Time:  ",time.time()-t_X)


# B = scale_variances(A,X)
# bayes_A = bayes_variance_graph(P,X,constants)
# bayes_B = scale_variances(bayes_A,X)
# bayes_dif = graph(adj_mat=bayes_B.adj-B.adj)
# # A.print()
# # print()
# # B.print()
# # print()
# # bayes_variance_graph(P,X,constants).print()
# # print()
# # bayes_B.print()
# # print()
# # bayes_dif.print()
# # print()
# print()
# print("True Var:  ",np.sum(B.adj))
# print("Bayes Var: ",np.sum(bayes_B.adj))
# print("Difference:",np.sum(np.abs(bayes_dif.adj)))
# print()
# S = np.array([[len(X[a,b][0]) for b in range(p)] for a in range(p)])
# print("Avg. Meas.:",sum([S[a,a] for a in range(p)])/shots)



rest = ["4qubits_H2","12qubits_LiH"]
for name in ["8qubits_H2","14qubits_BeH2","14qubits_H2O","16qubits_NH3","20qubits_C2","20qubits_HCl"]:
    for form in ["bk","jw","parity"]:
        print(name,form)
        shots = 1000

        file_Hamiltonian = "./Hamiltonians/"+name+"/"+form+".txt"
        file_ground_state = "./Hamiltonians/"+name+"/"+form+"_ground_state.txt"
        file_variance_graph = "./Hamiltonians/"+name+"/"+form+"_variance_graph.txt"
        file_maximal_cliques = "./Hamiltonians/"+name+"/"+form+"_maximal_cliques.txt"

        p,q,P,constants = read_Hamiltonian(file_Hamiltonian)
        print("Paulis:",p)
        print("Qubits:",q)

        t_psi = time.time()
        if os.path.isfile(file_ground_state):
            psi = read_ground_state(file_ground_state)
        else:
            psi = ground_state(P,constants)
            write_ground_state(file_ground_state,psi)
        print("Psi Time:",time.time()-t_psi)

        t_A = time.time()
        if os.path.isfile(file_variance_graph):
            A = read_variance_graph(file_variance_graph)
        else:
            A = variance_graph(P,constants,psi)
            write_variance_graph(file_variance_graph,A)
        print("A Time:  ",time.time()-t_A)

        t_aaa = time.time()
        # aaa = covering__cliques(maximalA,1)
        aaa = maximal_cliques(A)
        print("aaa Time:",time.time()-t_aaa)

        t_X = time.time()
        X = greedy_bayes_min_var(P,constants,psi,aaa,shots)
        # print(X)
        # print()
        print("X Time:  ",time.time()-t_X)

        bayes_A = bayes_variance_graph(P,X,constants)

        B = scale_variances(A,X)
        bayes_B = scale_variances(bayes_A,X)
        bayes_dif = graph(adj_mat=bayes_A.adj-A.adj)
        print("True Var:  ",np.sum(B.adj))
        print("Bayes Var: ",np.sum(bayes_B.adj))
        print("Difference:",np.sum(np.abs(bayes_dif.adj)))
        S = np.array([[len(X[a,b][0]) for b in range(p)] for a in range(p)])
        print("Avg. Meas.:",sum([S[a,a] for a in range(p)])/shots)
        print()





# shots = 100
# q = 10
# p = 150
# P = random_Ham(p,q,q)
# constants = [random.randrange(-1,1) for a in range(p)]
# psi = ground_state(P,constants)
# A = commutation_graph(P)
# aaa = maximal_cliques(A)
# t_X = time.time()
# aaa,aaa1 = itertools.tee(aaa,2)
# X = greedy_bayes_min_var(P,constants,psi,aaa1,shots)
# print(len(X[0,0][0]))
# # print(X)
# # print()
# print("X Time:  ",time.time()-t_X)
# t_X = time.time()
# aaa,aaa1 = itertools.tee(aaa,2)
# X = greedy_bayes_min_var2(A,P,constants,psi,aaa1,shots)
# print(len(X[0,0][0]))
# # print(X)
# # print()
# print("X Time:  ",time.time()-t_X)






