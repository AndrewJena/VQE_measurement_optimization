from Functions import *



# HAMILTONIAN NAMES

# "4qubits_H2"
# "8qubits_H2"
# "12qubits_LiH"
# "14qubits_BeH2"
# "14qubits_H2O"
# "16qubits_NH3"
# "20qubits_C2"
# "20qubits_HCl"



# HAMILTONIAN FORMS

# "bk"
# "jw"
# "parity"



# TEST CODE

# number of times to run each Hamiltonian
runs = 1

# number of shots to take in each simulation
shots = 1000

# when should variance graph update for better measurement allocations?
#     if step size = shots, the variance graph will never update
update_var = set(range(0,shots,1))

# part_func = LDF
# part_func = vertex_covering_maximal_cliques
part_func = all_maximal_cliques
print(part_func.__name__)

for name in ["4qubits_H2","8qubits_H2","12qubits_LiH","14qubits_BeH2","14qubits_H2O","16qubits_NH3","20qubits_C2","20qubits_HCl"]:
    for form in ["bk","jw","parity",]:
        print()
        print(name,form)

        file_Hamiltonian = "./Hamiltonians/"+name+"/"+form+".txt"
        file_ground_state = "./Hamiltonians/"+name+"/"+form+"_ground_state.txt"
        file_variance_graph = "./Hamiltonians/"+name+"/"+form+"_variance_graph.txt"

        p,q,P,constants = read_Hamiltonian(file_Hamiltonian)
        print("Paulis:",p)
        print("Qubits:",q)
        for _ in range(runs):

            t_psi = time.time()
            if os.path.isfile(file_ground_state):
                psi = read_ground_state(file_ground_state)
            else:
                psi = ground_state(P,constants)
                write_ground_state(file_ground_state,psi)
            # print("Psi Time:   ",f'{time.time()-t_psi:.10f}')

            t_A = time.time()
            if os.path.isfile(file_variance_graph):
                A = read_variance_graph(file_variance_graph)
            else:
                A = variance_graph(P,constants,psi)
                write_variance_graph(file_variance_graph,A)
            # print("A Time:     ",f'{time.time()-t_A:.10f}')

            t_X = time.time()
            X = greedy_bayes_min_var(P,constants,psi,shots,part_func,update_var=update_var)
            # print("X Time:     ",f'{time.time()-t_X:.10f}')

            bayes_A = bayes_variance_graph(X,constants)

            B = scale_variances(A,X)
            bayes_B = scale_variances(bayes_A,X)
            bayes_dif = graph(adj_mat=bayes_A.adj-A.adj)
            print("True Error: ",f'{np.sqrt(np.sum(B.adj)):.10f}')
            print("Bayes Error:",f'{np.sqrt(np.sum(bayes_B.adj)):.10f}')
            print("Difference: ",f'{np.sum(np.abs(bayes_dif.adj))/np.count_nonzero(commutation_graph(P).adj):.10f}')
            print("True Mean: ",f'{sum(constants[a]*Mean(P.a_pauli(a),psi) for a in range(p)):.10f}')
            print("Naive Mean:",f'{sum(constants[a]*naive_Mean(X[a,a]) for a in range(p)):.10f}')
            print()









