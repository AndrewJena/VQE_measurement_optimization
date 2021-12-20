from private_Functions import *



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

# "jw"
# "bk"
# "parity"





# TEST CODE


# set true for full simulation for estimated mean and error, or false for just true error and mean
full_simulation = False

# number of times to run each Hamiltonian
runs = 1

# number of shots to take in each simulation
shots = 1000

# when should variance graph update for better measurement allocations?
#     L=1 is a non-adaptive algorithm, whereas L>1 is adaptive
#     l=0 adapts at equal intervals, whereas l>0 updates closer to the beginning
L,l = 1,0
update_steps = Ll_updates(L,l,shots)
print(sorted(update_steps))

# which partition funciton should be used?
part_func = LDF
# part_func = all_maximal_cliques
# part_func = vertex_covering_maximal_cliques
# part_func = weighted_vertex_covering_maximal_cliques
print(part_func.__name__)
print()

unused_names = []
unused_forms = []
for name in ["4qubits_H2","8qubits_H2","12qubits_LiH","14qubits_BeH2","14qubits_H2O","16qubits_NH3","20qubits_C2","20qubits_HCl",]:
    for form in ["jw","bk","parity",]:
        print(name,form)

        file_Hamiltonian = "./Hamiltonians/"+name+"/"+form+".txt"
        file_ground_state = "./Hamiltonians/"+name+"/"+form+"_ground_state.txt"
        file_variance_graph = "./Hamiltonians/"+name+"/"+form+"_variance_graph.txt"

        Paulis,coefficients = read_Hamiltonian(file_Hamiltonian)
        p,q = Paulis.paulis(),Paulis.qubits()
        # print("Paulis:",p)
        # print("Qubits:",q)

        t_psi = time.time()
        if os.path.isfile(file_ground_state):
            psi = read_ground_state(file_ground_state)
        else:
            psi = ground_state(Paulis,coefficients)
            write_ground_state(file_ground_state,psi)
        # print("Psi Time:   ",f'{time.time()-t_psi:.10f}')

        t_A = time.time()
        if os.path.isfile(file_variance_graph):
            A = read_variance_graph(file_variance_graph)
        else:
            A = variance_graph(Paulis,coefficients,psi)
            write_variance_graph(file_variance_graph,A)
        # print("A Time:     ",f'{time.time()-t_A:.10f}')
        
        if full_simulation:
            t_X = time.time()
            XX = list(list(zip(*[bucket_filling(Paulis,coefficients,psi,shots,part_func,update_steps=update_steps,repeats=(i,runs),full_simulation=full_simulation) for i in range(runs)]))[1])
            # print("X Time:     ",f'{time.time()-t_X:.10f}')

            true_error = 0
            estim_error = 0
            estim_mean = 0
            for X in XX:
                bayes_A = bayes_variance_graph(X,coefficients)
                S = np.array([[sum(X[i0,i1].values()) for i1 in range(p)] for i0 in range(p)])
                true_error += np.sqrt(np.sum(scale_variances(A,S).adj))
                estim_error += np.sqrt(np.sum(scale_variances(bayes_A,S).adj))
                estim_mean += sum(coefficients[i]*naive_Mean(X[i,i]) for i in range(p))
            # "True Error" uses the real (hidden) variance matrix
            print("True Error: ",f'{true_error/runs:.10f}')
            # "Estim Error" uses the Bayesian estimte of the variance matrix
            print("Estim Error:",f'{estim_error/runs:.10f}')
            # "True Mean" is the true ground state of the Haamiltonian
            print("True Mean: ",f'{Hamiltonian_Mean(Paulis,coefficients,psi):.10f}')
            # "Estim Mean" is the estimated ground state given by a naive average after sampling
            print("Estim Mean:",f'{estim_mean/runs:.10f}')
            print()
        else:
            t_S = time.time()
            SS = list(list(zip(*[bucket_filling(Paulis,coefficients,psi,shots,part_func,update_steps=update_steps,repeats=(i,runs),full_simulation=full_simulation) for i in range(runs)]))[0])
            # print("S Time:     ",f'{time.time()-t_S:.10f}')

            true_error = 0
            for S in SS:
                true_error += np.sqrt(np.sum(scale_variances(A,S).adj))
            # "True Error" uses the real (hidden) variance matrix
            print("True Error: ",f'{true_error/runs:.10f}')
            # "True Mean" is the true ground state of the Haamiltonian
            print("True Mean: ",f'{Hamiltonian_Mean(Paulis,coefficients,psi):.10f}')
            print()







