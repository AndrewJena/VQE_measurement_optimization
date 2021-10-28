import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import itertools
import random
import time
import math
import functools
import os
import networkx as nx



# NAMING CONVENTIONS

# a - int
# aa - list{int}
# aaa - list{list{int}}
# b - bool
# c - float (constant)
# r - random
# i - indexing
# ss - str
# m - scipy.sparse.csr_matrix
# P - Pauli
# C - circuit
# G - gate
# A - graph (adjacency matrix)
# p - number of paulis
# q - number of qubits
# functions or methods which end with underscores modify the inputs



# DEFINITIONS

# I,X,Y,Z Pauli matrices used for constructing tensor products
I_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]],dtype=complex))
X_mat = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]],dtype=complex))
Y_mat = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]],dtype=complex))
Z_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]],dtype=complex))

# H,S Clifford matrices used for constructing Clifford gates
H_mat = scipy.sparse.csr_matrix(1/np.sqrt(2)*np.array([[1,1],[1,-1]],dtype=complex))
S_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,1j]],dtype=complex))



# READING, WRITING, AND MISCELLANEOUS

# read Hamiltonian from file, in Pauli string // (coefficient+0j) format
def read_Hamiltonian(path):
    # Inputs:
    #     path - (str) - path to Hamiltonian file
    # Outputs:
    #     (int)         - number of Paulis in Hamiltonian
    #     (int)         - number of qubits in Hamiltonian
    #     (pauli)       - set of Paulis in Hamiltonian
    #     (list{float}) - coefficients in Hamiltonian
    f = open(path,"r")
    ll = f.readlines()
    f.close()
    p = len(ll)//2
    q = len(ll[0])-1
    sss = []
    constants = []
    for a in range(0,len(ll),2):
        sss.append(ll[a][0:-1])
        constants.append(float(ll[a+1][1:-5]))
    return p,q,string_to_pauli(sss),constants

# write the ground state to a file for easy lookup
def write_ground_state(path,psi):
    # Inputs:
    #     path - (str)         - path to desired ground state file
    #     psi  - (numpy.array) - ground state of Hamiltonian
    ss = list(psi[a] for a in range(len(psi)))
    f = open(path,"w")
    for s in ss[:-1]:
        f.write(str(s)+" ")
    f.write(str(ss[-1]))
    f.close()

# read the ground state from a prepared file
def read_ground_state(path):
    # Inputs:
    #     path - (str) - path to ground state file
    # Outputs:
    #     (numpy.array) - ground state of Hamiltonian
    f = open(path,"r")
    ll = f.readlines()
    f.close()
    psi = []
    for l in ll:
        psi.append([np.complex(s) for s in l.split(" ")])
    return np.array(*psi)

# write the true variance graph to a file for easy lookup
def write_variance_graph(path,A):
    # Inputs:
    #     path - (str)   - path to desired variance graph file
    #     A    - (graph) - variance graph of Hamiltonian
    sss = list(list(str(A.adj[a0,a1]) for a1 in range(A.ord())) for a0 in range(A.ord()))
    f = open(path,"w")
    for ss in sss[:-1]:
        for s in ss[:-1]:
            f.write(s+" ")
        f.write(ss[-1]+"\n")
    for s in sss[-1][:-1]:
        f.write(s+" ")
    f.write(sss[-1][-1])
    f.close()

# read the true variance graph from a prepared file
def read_variance_graph(path):
    # Inputs:
    #     path - (str) - path to variance graph file
    # Outputs:
    #     (graph) - variance graph of Hamiltonian
    f = open(path,"r")
    ll = f.readlines()
    f.close()
    adj_mat = []
    for l in ll:
        adj_mat.append([float(s) for s in l.split(" ")])
    return graph(np.array(adj_mat))

# call within a loop to produce a loading bar which can be scaled for various timings
def loading_bar(a0,a1,length=50,scaling=lambda x : x):
    # Inputs:
    #     a0      - (int)      - current iteration
    #     a1      - (int)      - total iterations
    #     length  - (int)      - length of bar
    #     scaling - (function) - normalize speed of progress
    a2,a3 = scaling(a0),scaling(a1)
    ss = ("{0:.1f}").format(100*a2/a3)
    a4 = int(length*a2/a3)
    bar = '█'*a4 + '-'*(length-a4)
    print(f' |{bar}| {ss}% {a0}/{a1}', end="\r")
    if a0 == a1: 
        print(" "*(length+12+2*len(str(a1))),end="\r")



# PAULIS

# a class for storing sets of Pauli operators as pairs of symplectic matrices
class pauli:
    def __init__(self,X,Z):
        # Inputs:
        #     X - (numpy.array) - X-part of Pauli in symplectic form with shape (p,q)
        #     Z - (numpy.array) - Z-part of Pauli in symplectic form with shape (p,q)
        if X.shape != Z.shape:
            raise Exception("X- and Z-parts must have same shape")
        self.X = X
        self.Z = Z

    # check whether self has only X component
    def is_IX(self):
        # Outputs:
        #     (bool) - True if self has only X componenet, False otherwise
        if np.any(self.Z):
            return False
        return True

    # check whether self has only Z component 
    def is_IZ(self):
        # Outputs:
        #     (bool) - True if self has only Z componenet, False otherwise
        if np.any(self.X):
            return False
        return True

    # check whether the set of Paulis are pairwise commuting
    def is_commuting(self):
        # Outputs:
        #     (bool) - True if self is pairwise commuting set of Paulis
        p = self.paulis()
        PP = [self.a_pauli(a) for a in range(p)]
        return not any(symplectic_inner_product(PP[a0],PP[a1]) for a0,a1 in itertools.combinations(range(p),2))

    # check whether the set of Paulis are pairwise commuting on every qubit
    def is_qubitwise_commuting(self):
        # Outputs:
        #     (bool) - True if self is pairwise qubitwise commuting set of Paulis
        p = self.paulis()
        PP = [self.a_pauli(a) for a in range(p)]
        return not any(any((PP[a0].X[0,a2]&PP[a1].Z[0,a2])^(PP[a0].Z[0,a2]&PP[a1].X[0,a2]) for a2 in range(self.qubits())) for a0,a1 in itertools.combinations(range(p),2))

    # pull out the ath Pauli from self
    def a_pauli(self,a):
        # Inputs: 
        #     a - (int) - index of Pauli to be returned
        # Outputs:
        #     (pauli) - the ath Pauli in self
        return pauli(np.array([self.X[a,:]]),np.array([self.Z[a,:]]))

    # count the number of Paulis in self
    def paulis(self):
        # Output: (int)
        return self.X.shape[0]

    # count the number of qubits in self
    def qubits(self):
        # Outputs: (int)
        return self.X.shape[1]

    # delete Paulis indexed by aa
    def delete_paulis_(self,aa):
        # Inputs: 
        #     aa - (list of int)
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=0)
            self.Z = np.delete(self.Z,aa,axis=0)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=0)
                self.Z = np.delete(self.Z,a,axis=0)

    # return self after deletion of qubits indexed by aa
    def delete_qubits_(self,aa):
        # Inputs: 
        #     aa - (list of int)
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=1)
            self.Z = np.delete(self.Z,aa,axis=1)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=1)
                self.Z = np.delete(self.Z,a,axis=1)

    # return deep copy of self
    def copy(self):
        # Outputs: (pauli)
        X = np.array([[self.X[a0,a1] for a1 in range(self.qubits())] for a0 in range(self.paulis())],dtype=bool)
        Z = np.array([[self.Z[a0,a1] for a1 in range(self.qubits())] for a0 in range(self.paulis())],dtype=bool)
        return pauli(X,Z)

    # print string representation of self
    def print(self):
        sss = pauli_to_string(self)
        if type(sss) is str:
            print(sss)
        else:
            for ss in sss:
                print(ss)

    # print symplectic representation of self
    def print_symplectic(self):
        for a in range(self.paulis()):
            print(''.join(str(int(a1)) for a1 in self.X[a,:]),''.join(str(int(a1)) for a1 in self.Z[a,:]))

# convert a pauli object to its matrix representation
def pauli_to_matrix(P):
    # Inputs:
    #     P - (pauli) - must have shape (1,q)
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of input Pauli
    if P.paulis() != 1:
        raise Exception("Matrix can only be constructed for a single Pauli")
    X,Z = P.X[0],P.Z[0]
    mmdict = {(0,0):I_mat,(0,1):Z_mat,(1,0):X_mat,(1,1):Y_mat}
    return tensor([mmdict[(X[a],Z[a])] for a in range(P.qubits())])

# convert a collection of strings (or single string) to a pauli object
def string_to_pauli(sss):
    # Inputs:
    #     sss - (list{str}) or (str) - string representation of Pauli
    # Outputs:
    #     (pauli) - Pauli corresponding to input string(s)
    Xdict = {"I":0,"X":1,"Y":1,"Z":0}
    Zdict = {"I":0,"X":0,"Y":1,"Z":1}
    if type(sss) is str:
        X = np.array([[Xdict[s] for s in sss]],dtype=bool)
        Z = np.array([[Zdict[s] for s in sss]],dtype=bool)
        return pauli(X,Z)
    else:
        X = np.array([[Xdict[s] for s in ss] for ss in sss],dtype=bool)
        Z = np.array([[Zdict[s] for s in ss] for ss in sss],dtype=bool)
        return pauli(X,Z)

# convert a pauli object to a collection of strings (or single string)
def pauli_to_string(P):
    # Inputs:
    #     P - (pauli) - Pauli to be stringified
    # Outputs:
    #     (list{str}) - string representation of Pauli
    X,Z = P.X,P.Z
    ssdict = {(0,0):"I",(0,1):"Z",(1,0):"X",(1,1):"Y"}
    if P.paulis() == 0:
        return ''
    elif P.paulis() == 1:
        return ''.join(ssdict[(X[0,a],Z[0,a])] for a in range(P.qubits()))
    else:
        return [''.join(ssdict[(X[a0,a1],Z[a0,a1])] for a1 in range(P.qubits())) for a0 in range(P.paulis())]

# the symplectic inner product of two pauli objects (each with a single Pauli)
def symplectic_inner_product(P1,P2):
    # Inputs:
    #     P1 - (pauli) - must have shape (1,q)
    #     P2 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (int) - symplectic inner product of Paulis modulo 2
    if (P1.paulis() != 1) or (P2.paulis() != 1):
        raise Exception("Symplectic inner product only works with pair of single Paulis")
    if P1.qubits() != P2.qubits():
        raise Exception("Symplectic inner product only works if Paulis have same number of qubits")
    return functools.reduce(lambda x,y:x^y,np.logical_xor(np.logical_and(P1.X,P2.Z),np.logical_and(P1.Z,P2.X))[0,:])

# the symplectic inner product of two pauli objects (each with a single Pauli)
def qubitwise_inner_product(P1,P2):
    # Inputs:
    #     P1 - (pauli) - must have shape (1,q)
    #     P2 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (int) - qubitwise inner product of Paulis modulo 2
    if (P1.paulis() != 1) or (P2.paulis() != 1):
        raise Exception("Qubitwise inner product only works with pair of single Paulis")
    if P1.qubits() != P2.qubits():
        raise Exception("Qubitwise inner product only works if Paulis have same number of qubits")
    return any((P1.X[0,a]&P2.Z[0,a])^(P1.Z[0,a]&P2.X[0,a]) for a in range(P1.qubits()))

# the product of two pauli objects
def pauli_product(P1,P2):
    # Inputs:
    #     P1 - (pauli) - must have shape (1,q)
    #     P2 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (pauli) - product of Paulis
    if P1.paulis() != 1 or P2.paulis() != 1:
        raise Exception("Product can only be calculated for single Paulis")
    return pauli(np.logical_xor(P1.X,P2.X),np.logical_xor(P1.Z,P2.Z))



# GATES & CIRCUITS

# a class for storing quantum gates as a name and a set of qubits
class gate:
    def __init__(self,name,aa):
        # Inputs:
        #     name - (function)  - function for action of gate
        #     aa   - (list{int}) - list of qubits
        self.name = name
        self.aa = aa

    # returns the name of the gate as a string
    def name_string(self):
        # Outputs:
        #     (str) - name of function as a string
        return self.name.__name__

    # deep copy of self
    def copy(self):
        # Outputs:
        #     (gate) - deep copy of gate
        return gate(self.name,[a for a in self.aa])

    # print self as a name and a tuple of qubits
    def print(self):
        print("%s(%s)"%(self.name_string(),str(self.aa).replace(' ','')[1:-1]))

# a class for storing quantum circuits as a collection of gates along with the dimension
class circuit:
    def __init__(self,q):
        # Inputs:
        #     qubits - (int) - number of qubits in circuit
        self.qubits = q
        self.gg = []

    # the number of gates in the circuit
    def length(self):
        # Outputs:
        #     (int) - number of gates in circuit
        return len(self.gg)

    # convert self to its corresponding unitary matrix
    def unitary(self):
        # Outputs:
        #     (scipy.sparse.csr_matrix) - unitary matrix representation of self
        m = scipy.sparse.csr_matrix(([1]*(1<<self.qubits),(range(1<<self.qubits),range(1<<self.qubits))))
        for g in self.gg:
            m = globals()[g.name_string()+'_unitary'](g.aa,self.qubits) @ m
        return m

    # append gates to the end of self
    def add_gates_(self,C):
        # Inputs:
        #     C - (circuit) or (list{gate}) or (gate) - gates to be appended to self
        if type(C) is circuit:
            self.gg += C.gg
        elif type(C) is gate:
            self.gg.append(C)
        else:
            self.gg += C

    # insert gates at a given timestep in self
    def insert_gates_(self,C,a):
        # Inputs:
        #     C - (circuit) or (list{gate}) or (gate) - gates to be inserted into self
        #     a  - (int)        - index for insertion
        if type(C) is gate:
            self.gg.insert(a,C)
        elif type(C) is circuit:
            self.gg[a:a] = C.gg
        else:
            self.gg[a:a] = C

    # delete gates at specific timesteps
    def delete_gates_(self,aa):
        # Inputs:
        #     aa - (int) or (list{int}) - indices where gates should be deleted
        if type(aa) is int:
            del self.gg[aa]
        else:
            self.gg = [self.gg[a1] for a1 in range(self.length()) if not a1 in aa]

    # deep copy of self
    def copy(self):
        # Outputs:
        #     (circuit) - deep copy of self
        return circuit(self.qubits,[g.copy() for g in self.gg])

    # print self as gates on consecutive lines
    def print(self):
        for g in self.gg:
            g.print()

# returns the outcome of a given circuit (or gate or list of gates) acting on a given pauli
def act(P,C):
    # Inputs:
    #     P - (pauli)                             - Pauli to be acted upon
    #     C - (gate) or (circuit) or (list{gate}) - gates to act on Pauli
    # Outputs:
    #     (pauli) - result of C acting on P by conjugation
    if P == None:
        return P
    elif type(C) is gate:
        return C.name(P,C.aa)
    elif type(C) is circuit:
        return act(P,C.gg)
    elif len(C) == 0:
        return P
    elif len(C) == 1:
        return act(P,C[0])
    else:
        return act(act(P,C[0]),C[1:])

# function for the gate representation of Hadamard gate
def H(P,aa):
    # Inputs:
    #     P  - (pauli)     - Pauli to be acted upon
    #     aa - (list{int}) - qubits to be acted upon
    # Outputs:
    #     (pauli) - result of H(aa) acting on P
    # X -> Z
    # Z -> X
    Q = P.copy()
    X,Z = Q.X,Q.Z
    for a in aa:
        X[:,a],Z[:,a] = Z[:,a].copy(),X[:,a].copy()
    return pauli(X,Z)

# function for the gate representation of phase gate
def S(P,aa):
    # Inputs:
    #     P  - (pauli)     - Pauli to be acted upon
    #     aa - (list{int}) - qubits to be acted upon
    # Outputs:
    #     (pauli) - result of S(aa) acting on P
    # X -> Y
    # Z -> Z
    Q = P.copy()
    X,Z = Q.X,Q.Z
    for a in aa:
        Z[:,a] ^= X[:,a]
    return pauli(X,Z)

# function for the gate representation of CNOT gate
def CX(P,aa):
    # Inputs:
    #     P  - (pauli)     - Pauli to be acted upon
    #     aa - (list{int}) - control aa[0] and target aa[1]
    # Outputs:
    #     (pauli) - result of CNOT(aa[0],aa[1]) acting on P
    # XI -> XX
    # IX -> IX
    # ZI -> ZI
    # IZ -> ZZ
    Q = P.copy()
    X,Z = Q.X,Q.Z
    a0,a1 = aa[0],aa[1]
    X[:,a1] ^= X[:,a0]
    Z[:,a0] ^= Z[:,a1]
    return pauli(X,Z)

# function for the gate representation of CZ gate
def CZ(P,aa):
    # Inputs:
    #     P  - (pauli)     - Pauli to be acted upon
    #     aa - (list{int}) - targets aa[0] and aa[1]
    # Outputs:
    #     (pauli) - result of CZ(aa[0],aa[1]) acting on P
    # XI -> XZ
    # IX -> ZX
    # ZI -> ZI
    # IZ -> IZ
    X,Z = P.X,P.Z
    a0,a1 = aa[0],aa[1]
    Z[:,a0] ^= X[:,a1]
    Z[:,a1] ^= X[:,a0]
    return pauli(X,Z)

# function for the gate representation of SWAP gate
def SWAP(P,aa):
    # Inputs:
    #     P  - (pauli)     - Pauli to be acted upon
    #     aa - (list{int}) - targets aa[0] and aa[1]
    # Outputs:
    #     (pauli) - result of SWAP(aa[0],aa[1]) acting on P
    # XI -> IX
    # IX -> XI
    # ZI -> IZ
    # IZ -> ZI
    X,Z = P.X,P.Z
    a0,a1 = aa[0],aa[1]
    X[:,a0],X[:,a1] = X[:,a1].copy(),X[:,a0].copy()
    Z[:,a0],Z[:,a1] = Z[:,a1].copy(),Z[:,a0].copy()
    return pauli(X,Z)

# function for mapping Hadamard gate to corresponding unitary matrix
def H_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - indices for Hadamard tensors
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional H(aa)
    return tensor([H_mat if a in aa else I_mat for a in range(q)])

# function for mapping phase gate to corresponding unitary matrix
def S_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - indices for phase gate tensors
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional S(aa)
    return tensor([S_mat if a in aa else I_mat for a in range(q)])

# function for mapping CNOT gate to corresponding unitary matrix
def CX_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - control aa[0] and target aa[1]
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional CNOT(aa[0],aa[1])
    a0 = q-1-aa[0]
    a1 = q-1-aa[1]
    aa2 = np.array([1 for a2 in range(1<<q)])
    aa3 = np.array([a3 for a3 in range(1<<q)])
    aa4 = np.array([(a4^((a4&(1<<(a0)))>>a0)*(1<<(a1))) for a4 in range(1<<q)])
    return scipy.sparse.csr_matrix((aa2,(aa3,aa4)))

# function for mapping CZ gate to corresponding unitary matrix
def CZ_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - targets aa[0] and aa[1]
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional CZ(aa[0],aa[1])
    a0 = q-1-aa[0]
    a1 = q-1-aa[1]
    aa2 = np.array([(-1)**(((a2&(1<<(a0)))>>a0)&((a2&(1<<(a1)))>>a1)) for a2 in range(1<<q)])
    aa3 = np.array([a3 for a3 in range(1<<q)])
    aa4 = np.array([a4 for a4 in range(1<<q)])
    return scipy.sparse.csr_matrix((aa2,(aa3,aa4)))

# function for mapping SWAP gate to corresponding unitary matrix
def SWAP_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - targets aa[0] and aa[1]
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional SWAP(aa[0],aa[1])
    a0 = q-1-aa[0]
    a1 = q-1-aa[1]
    aa2 = np.array([1 for a2 in range(1<<q)])
    aa3 = np.array([a3 for a3 in range(1<<q)])
    aa4 = np.array([a4^(1<<a0)^(1<<a1) if (((a4&(1<<a0))>>a0)^((aa4&(1<<a1))>>a1)) else a4 for a4 in range(1<<q)])
    return scipy.sparse.csr_matrix((aa2,(aa3,aa4)))

# returns the circuit which diagonalizes a pairwise commuting pauli object
def diagonalize(P):
    # Inputs:
    #     P - (pauli) - Pauli to be diagonalized
    # Outputs:
    #     (circuit) - circuit which diagonalizes P
    q = P.qubits()
    # if not P.is_commuting():
    #     raise Exception("Paulis must be pairwise commuting to be diagonalized")
    P1 = P.copy()
    C = circuit(q)
    # for each qubit, call diagonalize_iter_
    for a in range(q):
        C = diagonalize_iter_(P1,C,a)
    P1 = act(P1,C)
    # if any qubits are X rather than Z, apply H to make them Z
    if [a for a in range(P1.qubits()) if any(P1.X[:,a])]:
        C.add_gates_(gate(H,[a for a in range(q) if any(P1.X[:,a])]))
    return C

# an iterative function called within diagonalize()
def diagonalize_iter_(P,C,a):
    # Inputs:
    #     P - (pauli)   - Pauli to be diagonalized
    #     C - (circuit) - circuit which diagonalizes first a-1 qubits
    #     a - (int)     - current qubit
    # Outputs:
    #     (circuit) - circuit which diagonalizes first a qubits
    p,q = P.paulis(),P.qubits()
    P = act(P,C)

    # if all Paulis have no X-part on qubit a, return C
    if not any(P.X[:,a]):
        return C

    # set a1 to be the index of the minimum Pauli with non-zero X-part of qubit a
    a1 = min(a2 for a2 in range(p) if P.X[a2,a])

    # add CNOT gates to cancel out all non-zero X-parts on Pauli a1, qubits > a
    if any(P.X[a1,a2] for a2 in range(a+1,q)):
        gg = [gate(CX,[a,a2]) for a2 in range(a+1,q) if P.X[a1,a2]]
        C.add_gates_(gg)
        P = act(P,gg)

    # check whether there are any non-zero Z-parts on Pauli a1, qubits > a
    if any(P.Z[a1,a2] for a2 in range(a+1,q)):

        # if Pauli a1, qubit a is X, apply S gate to make it Y
        if not P.Z[a1,a]:
            g = gate(S,[a])
            C.add_gates_(g)
            P = act(P,g)

        # add backwards CNOT gates to cancel out all non-zero Z-parts on Pauli a1, qubits > a
        gg = [gate(CX,[a2,a]) for a2 in range(a+1,q) if P.Z[a1,a2]]
        C.add_gates_(gg)
        P = act(P,gg)

    # if Pauli a1, qubit a is Y, add S gate to make it X
    if P.Z[a1,a]:
        g = gate(S,[a])
        C.add_gates_(g)
        P = act(P,g)
    return C



# GRAPHS

# a class for storing graphs as adjacency matrices
#     since we are dealing with covariance matrices with both vertex and edge weights,
#     this is a suitable format to capture that complexity
class graph:
    # Inputs:
    #     adj_mat - (numpy.array) - (weighted) adjacency matrix of graph
    #     dtype   - (numpy.dtype) - data type of graph weights
    def __init__(self,adj_mat=np.array([]),dtype=float):
        self.adj = adj_mat.astype(dtype)

    # adds a vertex to self
    def add_vertex_(self,weight=1):
        # Inputs:
        #     weight - (float) - vertex weight
        if len(self.adj) == 0:
            self.adj = np.array([weight])
        else:
            r = np.zeros((1,len(self.adj)))
            c = np.zeros((len(self.adj),1))
            w = np.array([[weight]])
            self.adj = np.block([[self.adj,c],[r,w]])

    # weight a vertex
    def lade_vertex_(self,a,weight):
        # Inputs:
        #     a      - (int)   - vertex to be weighted
        #     weight - (float) - vertex weight
        self.adj[a,a] = weight

    # weight an edge
    def lade_edge_(self,a1,a2,weight):
        # Inputs:
        #     a1     - (int)   - first vertex
        #     a2     - (int)   - second vertex
        #     weight - (float) - vertex weight
        self.adj[a1,a2] = weight
        self.adj[a2,a1] = weight

    # returns a set of the neighbors of a given vertex
    def neighbors(self,a):
        # Inputs:
        #     a - (int) - vertex for which neighbors should be returned
        # Outputs:
        #     (list{int}) - set of neighbors of vertex a
        aa1 = set([])
        for a1 in range(self.ord()):
            if (a != a1) and (self.adj[a,a1] != 0):
                aa1.add(a1)
        return aa1

    # returns list of all edges in self
    def edges(self):
        # Outputs:
        #     (list{list{int}}) - list of edges in self
        aaa = []
        for a0,a1 in itertools.combinations(range(self.ord()),2):
            if a1 in self.neighbors(a0):
                aaa.append([v,w])
        return aaa

    # check whether a collection of vertices is a clique in self
    def clique(self,aa):
        # Inputs:
        #     aa - (list{int}) - list of vertices to be checked for clique
        # Outputs:
        #     (bool) - True if aa is a clique in self; False otherwise
        for a0,a1 in itertools.combinations(vv,2):
            if self.adj[a0,a1] == 0:
                return False
        return True

    # returns the degree of a given vertex
    def degree(self,a):
        # Inputs:
        #     a - (int) - vertex for which degree should be returned
        # Outputs:
        #     (int) - degree of vertex a
        return np.count_nonzero(self.adj[a,:])

    # returns the number of vertices in self
    def ord(self):
        # Outputs:
        #     (int) - number of vertices in self
        return self.adj.shape[0]

    # print adjacency matrix representation of self
    def print(self):
        for a in range(self.ord()):
            print('[',end=' ')
            for b in range(self.ord()):
                s = self.adj[a,b]
                if str(s)[0] == '-':
                    print(f'{self.adj[a,b]:.6f}',end=" ")
                else:
                    print(' '+f'{self.adj[a,b]:.6f}',end=" ")
            print(']')

    # print self as a list of vertices together with their neighbors
    def print_neighbors(self):
        for v in range(self.ord()):
            print(v,end=": ")
            for w in self.neighbors(v):
                print(w,end=" ")
            print()

    # return a deep copy of self
    def copy(self):
        # Outputs:
        #     (graph) - deep copy of self
        return graph(np.array([[self.adj[a,b] for b in range(self.ord())] for a in range(self.ord)]))

# returns all non-empty cliques in a graph
def nonempty_cliques(A):
    # Inputs:
    #     A - (graph) - graph for which all cliques should be found
    # Outputs:
    #     (list{list{int}}) - a list containing all non-empty cliques in A
    p = A.ord()
    aaa = set([frozenset([])])
    for a in range(p):
        aset = set([a])
        inter = A.neighbors(a)
        aaa |= set([frozenset(aset|(inter&aa)) for aa in aaa])
    aaa.remove(frozenset([]))
    return list([list(aa) for aa in aaa])

# returns an generator over all maximal cliques in a graph
def all_maximal_cliques(A):
    # Inputs:
    #     A - (graph) - graph for which all cliques should be found
    # Outputs:
    #     (generator) - a generator over all maximal cliques in A
    G = nx.Graph()
    G.add_nodes_from([a for a in range(A.ord())])
    G.add_edges_from([(a,b) for a in range(A.ord()) for b in A.neighbors(a)])
    return nx.algorithms.clique.find_cliques(G)

# returns a clique-covering of a graph which hits every vertex at least a certain number of times
def vertex_covering_maximal_cliques(A,k=1):
    # Inputs:
    #     A - (graph) - graph for which covering should be found
    #     k - (int)   - number of times each vertex should be covered
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which cover A
    p = A.ord()
    N = {}
    for a in range(p):
        N[a] = A.neighbors(a)
    aaa = []
    for b in range(p):
        for _ in range(k):
            cc = [b]
            dd = list(N[b])
            while dd:
                ww = [len(N[d].intersection(dd)) for d in dd]
                if sum(ww) == 0:
                    ww = [1 for d in dd]
                c = random.choices(dd,ww)[0]
                cc.append(c)
                dd = list(N[c].intersection(dd))
            aaa.append(cc)
    return aaa

# returns a largest degree first clique partition of a graph
def LDF(A):
    # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    p = A.ord()
    remaining = set(range(p))
    N = {}
    for a in range(p):
        N[a] = A.neighbors(a)
    aaa = []
    while remaining:
        a = max(remaining,key=lambda x : len(N[x]&remaining))
        aa = set([a])
        aa1 = N[a]&remaining
        while aa1:
            a2 = max(aa1,key=lambda x : len(N[x]&aa1))
            aa.add(a2)
            aa1 &= N[a2]
        aaa.append(aa)
        remaining -= aa
    return [sorted(list(aa)) for aa in aaa]



# PHYSICS FUNCTIONS

# returns the tensor product of a list of matrices
def tensor(mm):
    # Inputs:
    #     mm - (list{scipy.sparse.csr_matrix}) - matrices to tensor
    # Outputs:
    #     (scipy.sparse.csr_matrix) - tensor product of matrices
    if len(mm) == 0:
        return matrix([])
    elif len(mm) == 1:
        return mm[0]
    else:
        return scipy.sparse.kron(mm[0],tensor(mm[1:]),format="csr")

# returns the mean of a single Pauli with a given state
def Mean(P,psi):
    # Inputs:
    #     P   - (pauli)       - Pauli for mean
    #     psi - (numpy.array) - state for mean
    # Outputs:
    #     (numpy.float64) - mean <psi|P|psi>
    m = pauli_to_matrix(P)
    psi_dag = psi.conj().T
    mean = psi_dag @ m @ psi
    return mean.real

# returns the variance of a single Pauli with a given state
def Var(P,psi):
    # Inputs:
    #     P   - (pauli)       - Pauli for variance
    #     psi - (numpy.array) - state for variance
    # Outputs:
    #     (numpy.float64) - variance <psi|P^2|psi> - <psi|P|psi>^2
    # Output: 
    m = pauli_to_matrix(P)
    psi_dag = psi.conj().T
    var = (psi_dag @ m @ m @ psi)-(psi_dag @ m @ psi)**2
    return var.real

# returns the variance of two single Paulis with a given state
def Cov(P1,P2,psi):
    # Inputs:
    #     P1  - (pauli)       - first Pauli for covariance
    #     P2  - (pauli)       - second Pauli for covariance
    #     psi - (numpy.array) - state for variance
    # Outputs:
    #     (numpy.float64) - covariance <psi|P1P2|psi> - <psi|P1|psi><psi|P2|psi>
    m1 = pauli_to_matrix(P1)
    m2 = pauli_to_matrix(P2)
    psi_dag = psi.conj().T
    cov = (psi_dag @ m1 @ m2 @ psi)-(psi_dag @ m1 @ psi)*(psi_dag @ m2 @ psi)
    return cov.real

# returns the graph of variances and covariances for a given Hamiltonian and ground state
def variance_graph(P,constants,psi):
    # Inputs:
    #     P         - (pauli)         - set of Paulis in Hamiltonian
    #     constants - (list{float64}) - constants in Hamiltonian
    #     psi       - (numpy.array)   - ground state
    # Outputs:
    #     (graph) - variances and covariances of all Paulis with respect to ground state
    p = P.paulis()
    mm = [pauli_to_matrix(P.a_pauli(a)) for a in range(p)]
    psi_dag = psi.conj().T
    pmp = [(psi_dag@mm[a]@psi).real for a in range(p)]
    V = np.array([[constants[a1]*constants[a0]*((psi_dag@mm[a1]@mm[a0]@psi)-pmp[a1]*pmp[a0]).real for a1 in range(p)] for a0 in range(p)])
    return graph(adj_mat=V)

# returns the commutation graph of a given Pauli
def commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for commutation relations
    # Outputs:
    #     (graph) - an edge is weighted 1 if the pair of Paulis commute
    p = P.paulis()
    V = np.array([[1-symplectic_inner_product(P.a_pauli(a0),P.a_pauli(a1)) for a1 in range(p)] for a0 in range(p)])
    return graph(V,dtype=bool)

# returns the complement of the commutation graph
def anticommutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for anticommutation relations
    # Outputs:
    #     (graph) - edge is weighted 1 if the pair of Paulis anticommute
    p = P.paulis()
    V = np.array([[symplectic_inner_product(P.a_pauli(a0),P.a_pauli(a1)) for a1 in range(p)] for a0 in range(p)])
    return graph(adj_mat=V,dtype=bool)

# returns the qubitwise commutation graph of a given Pauli
def qubitwise_commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for qubitwise commutation relations
    # Outputs:
    #     (graph) - an edge is weighted 1 if the pair of Paulis qubitwise commute
    p = P.paulis()
    V = np.array([[1-qubitwise_inner_product(P.a_pauli(a0),P.a_pauli(a1)) for a1 in range(p)] for a0 in range(p)])
    return graph(adj_mat=V,dtype=bool)

# returns a random Hamiltonian with given number of Paulis, number of qubits, and Pauli weight
def random_Ham(p,q,d):
    # Inputs:
    #     p - (int) - number of Paulis
    #     q - (int) - number of qubits
    #     d - (int) - max Pauli weight
    # Outputs:
    #     (pauli) - random set of Paulis satisfying input conditions
    sss = []
    ssdict = {0:"I",1:"Z",2:"X",3:"Y"}
    for a in range(p):
        sss.append("".join([ssdict[random.randint(0,3)] if a1 in random.sample(range(q),d) else "I" for a1 in range(q)]))
    return string_to_pauli(sss)

# print list of Paulis in string form, together with constants
def print_Ham_string(P,constants):
    # Inputs:
    #     P         - (pauli)     - Pauli to be printed
    #     constants - (list{int}) - constants for Hamiltonian
    X,Z = P.X,P.Z
    for a in range(P.paulis()):
        print(pauli_to_string(P.a_pauli(a)),end="")
        if constants[a] >= 0:
            print(" +%s"%constants[a])
        else:
            print(" %s"%constants[a])

# returns the ground state of a given Hamiltonian
def ground_state(P,constants):
    # Inputs:
    #     P         - (pauli)     - Paulis for Hamiltonian
    #     constants - (list{int}) - constants for Hamiltonian
    # Outputs:
    #     (numpy.array) - eigenvector corresponding to lowest eigenvalue of Hamiltonian
    m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(P.paulis()))
    gval,gvec = scipy.sparse.linalg.eigsh(m,which='SA',k=1)
    return np.array([g for g in gvec[:,0]])



# MEASUREMENT FUNCTIONS

# some Paulis anticommute with some gates; we track those negations for a given Pauli and circuit
def negations(P,C):
    # Inputs:
    #     P - (pauli)   - set of Paulis for consideration
    #     C - (circuit) - circuit for consideration
    # Outputs:
    #     (list{bool}) - ith element is True if Pauli i picks up a negative sign, False otherwise
    Q = P.copy()
    bb = [False]*P.paulis()
    for g in C.gg:
        bb = np.logical_xor(bb,negation_iter(Q,g))
        Q = act(Q,g)
    return bb

# an iterative function for use in negations()
def negation_iter(P,g):
    # Inputs:
    #     P - (pauli) - set of Paulis for consideration
    #     g - (gate)  - gate for consideration
    # Outputs:
    #     (list{bool}) - ith element is True if Pauli i anticommutes with gate, False otherwise
    if g.name == S:
        return np.logical_xor.reduce([P.X[:,a1]&P.Z[:,a1] for a1 in g.aa])
    elif g.name == H:
        return np.logical_xor.reduce([P.X[:,a1]&P.Z[:,a1] for a1 in g.aa])
    elif g.name == CX:
        a0,a1 = g.aa
        return P.X[:,a0]&P.Z[:,a1]&(P.Z[:,a0]==P.X[:,a1])
    return [False]*P.paulis()

# samples ground state with respect to eigenstates of a clique
def sample_and_remember(P,psi,aa):
    # Inputs:
    #     P   - (pauli)     - Pauli for consideration
    #     psi - (circuit)   - circuit for consideration
    #     aa  - (list{int}) - clique to be measured
    # Outputs:
    #     (list{int}) - ith element eigenvalue of sampled eigenstate with respect to Pauli i
    P1 = P.copy()
    P1.delete_paulis_([a1 for a1 in range(P.paulis()) if not a1 in aa])
    C = diagonalize(P1)
    bb = negations(P1,C)
    P1 = act(P1,C)
    psi_diag = C.unitary() @ psi
    cdf = [0]
    for a1 in range(len(psi_diag)):
        cdf.append(cdf[-1]+np.absolute(psi_diag[a1])**2)
    l = len(cdf)
    p,q = P1.paulis(),P1.qubits()
    r = random.uniform(0,cdf[-1])
    a = max(np.where([c<=r for c in cdf])[0])
    sample = np.array([[a&(l>>(a1+1)) for a1 in range(q)]],dtype=bool)
    return [(-1)**(bb[a1]^functools.reduce(lambda x,y:x^y,(sample&P1.Z)[a1,:])) for a1 in range(p)],P1,cdf,bb

# samples ground state with respect to eigenstates of a clique using previously computed cdf
def sample_from_memory(P,cdf,bb):
    l = len(cdf)
    p,q = P.paulis(),P.qubits()
    r = random.uniform(0,cdf[-1])
    a = max(np.where([c<=r for c in cdf])[0])
    sample = np.array([[a&(l>>(b+1)) for b in range(q)]],dtype=bool)
    return [(-1)**(bb[a1]^functools.reduce(lambda x,y:x^y,(sample&P.Z)[a1,:])) for a1 in range(p)]

# scales the entries in a variance graph with respect to number of measurements
def scale_variances(A,X):
    # Inputs:
    #     A - (graph)             - variance matrix
    #     X - (numpy.array{Dict}) - array for tracking measurement outcomes
    p = A.ord()
    S0 = np.array([[sum(X[a0,a1].values()) for a1 in range(p)] for a0 in range(p)])
    S1 = np.array([[S0[a0,a1]/(S0[a0,a0]*S0[a1,a1]) if S0[a0,a1]>0 else int(a0==a1) for a1 in range(p)] for a0 in range(p)])
    return graph(np.multiply(S1,A.adj))



# ESTIMATED PHYSICS FUNCTIONS

# Bayesian estimation of mean from samples
def bayes_Mean(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    return (x0-x1)/(x0+x1+2)

# Bayesian estimation of variance from samples
def bayes_Var(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian variance of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    return 4*((x0+1)*(x1+1))/((x0+x1+2)*(x0+x1+3))

# Bayesian estimation of covariance from samples
def bayes_Cov(xyDict,xDict,yDict):
    # Inputs:
    #     xyDict - (Dict) - number of ++/+-/-+/-- outcomes for pair of Paulis
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for first Pauli
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for second Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    xy00,xy01,xy10,xy11 = xyDict[(1,1)],xyDict[(1,-1)],xyDict[(-1,1)],xyDict[(-1,-1)]
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    y0,y1 = yDict[(1,1)],yDict[(-1,-1)]
    p00 = 4*((x0+1)*(y0+1))/((x0+x1+2)*(y0+y1+2))
    p01 = 4*((x0+1)*(y1+1))/((x0+x1+2)*(y0+y1+2))
    p10 = 4*((x1+1)*(y0+1))/((x0+x1+2)*(y0+y1+2))
    p11 = 4*((x1+1)*(y1+1))/((x0+x1+2)*(y0+y1+2))
    return 4*((xy00+p00)*(xy11+p11) - (xy01+p01)*(xy10+p10))/((xy00+xy01+xy10+xy11+4)*(xy00+xy01+xy10+xy11+5))

# approximates the variance graph using Bayesian estimates
def bayes_variance_graph(X,constants):
    # Inputs:
    #     X         - (numpy.array{Dict}) - array for tracking measurement outcomes
    #     constants - (list{float})       - coefficients of Hamiltonian
    return graph(np.array([[(constants[a]**2)*bayes_Var(X[a,a]) if a==b else constants[a]*constants[b]*bayes_Cov(X[a,b],X[a,a],X[b,b]) for b in range(len(constants))] for a in range(len(constants))]))

# naive estimation of mean from samples
def naive_Mean(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    if (x0+x1) == 0:
        return 0
    return (x0-x1)/(x0+x1)

# naive estimation of variance from samples
def naive_Var(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian variance of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    if (x0+x1) == 0:
        return 2/3
    return 4*(x0*x1)/((x0+x1)*(x0+x1))

# naive estimation of covariance from samples
def naive_Cov(xyDict,xDict,yDict):
    # Inputs:
    #     xyDict - (Dict) - number of ++/+-/-+/-- outcomes for pair of Paulis
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for first Pauli
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for second Pauli
    # Outputs:
    #     (float) - naive estimate of mean
    xy00,xy01,xy10,xy11 = xyDict[(1,1)],xyDict[(1,-1)],xyDict[(-1,1)],xyDict[(-1,-1)]
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    y0,y1 = yDict[(1,1)],yDict[(-1,-1)]
    if (xy00+xy01+xy10+xy11) == 0:
        return 0
    return 4*((xy00)*(xy11) - (xy01)*(xy10))/((xy00+xy01+xy10+xy11)*(xy00+xy01+xy10+xy11))

# approximates the variance graph using naive estimates
def naive_variance_graph(X,constants):
    # Inputs:
    #     X         - (numpy.array{Dict}) - array for tracking measurement outcomes
    #     constants - (list{float})       - coefficients of Hamiltonian
    return graph(np.array([[(constants[a0]**2)*naive_Var(X[a0,a0]) if a0==a1 else constants[a0]*constants[a1]*naive_Cov(X[a0,a1],X[a0,a0],X[a1,a1]) for a1 in range(len(constants))] for a0 in range(len(constants))]))



# SIMULATION ALGORITHMS

# partitions Hamiltonian and repeatedly samples cliques while minimizing total variance
#     returns an array of dictionaries which tracks ++/+-/-+/-- outcomes for each pair of Paulis
def greedy_bayes_min_var(P,constants,psi,shots,part_func,update_var=1):
    # Inputs:
    #     P          - (pauli)       - Paulis in Hamiltonian
    #     constants  - (list{int})   - coefficients in Hamiltonian
    #     psi        - (numpy.array) - ground state of Hamiltonian
    #     shots      - (int)         - number of samples to take
    #     part_func  - (function)    - function for determining partition
    #     update_var - (int)         - number of turns between updating variance estimates
    # Outputs:
    #     (numpy.array{Dict}) - array of measurement outcome counts
    if part_func == None:
        return non_partitioning_bayes_min_var(P,constants,psi,shots,update_var=update_var)
    # if qubitwise commutation is preferred so no entangling gates are required, use the qubitwise commutation graph here
    aaa = part_func(commutation_graph(P))
    aaa,aaa1,aaa2 = itertools.tee(aaa,3)
    if not update_var&set(range(1,shots)) and not any(set(aa1)&set(aa2) for aa1,aa2 in itertools.product(aaa1,aaa2) if not aa1==aa2):
        return non_overlapping_bayes_min_var(P,constants,psi,shots,aaa)
    p = P.paulis()
    cdf_dict = {}
    X = np.array([[{(1,1):0,(1,-1):0,(-1,1):0,(-1,-1):0} for a1 in range(p)] for a0 in range(p)])
    S0 = np.array([[0 for a1 in range(p)] for a0 in range(p)],dtype=int)
    S1 = np.array([[1 for a1 in range(p)] for a0 in range(p)],dtype=float)
    V = np.array([[bayes_Var(X[a0,a0]) if a0==a1 else bayes_Cov(X[a0,a1],X[a0,a0],X[a1,a1]) for a1 in range(p)] for a0 in range(p)])
    U = np.array([[constants[a0]*constants[a1]*V[a0,a1]*S1[a0,a1] for a1 in range(p)] for a0 in range(p)])
    loading_bar(0,shots)
    for a2 in range(shots):
        aaa,aaa1 = itertools.tee(aaa,2)
        aa = sorted(max(aaa1,key=lambda x : U[x][:,x].sum()))
        if str(aa) not in cdf_dict.keys():
            mm,P1,cdf,bb = sample_and_remember(P,psi,aa)
            cdf_dict[str(aa)] = (P1,cdf,bb)
        else:
            mm = sample_from_memory(*cdf_dict[str(aa)])
        for (a0,m0),(a1,m1) in itertools.product(zip(aa,mm),repeat=2):
            X[a0,a1][(m0,m1)] += 1
        S0[np.ix_(aa,aa)] = np.add(S0[aa][:,aa],np.ones((len(aa),len(aa))))
        for a0,a1 in itertools.product(range(p),aa):
            if S0[a0,a1] > 0:
                r = (S0[a0,a1])/(S0[a0,a0]*S0[a1,a1])-(S0[a0,a1]+1)/((S0[a0,a0]+1)*(S0[a1,a1]+1))
                S1[a0,a1] = r
                S1[a1,a0] = r
        if (a2+1) in update_var:
            V = np.array([[bayes_Var(X[a0,a0]) if a0==a1 else bayes_Cov(X[a0,a1],X[a0,a0],X[a1,a1]) for a1 in range(p)] for a0 in range(p)])
            U = np.array([[constants[a0]*constants[a1]*V[a0,a1]*S1[a0,a1] for a1 in range(p)] for a0 in range(p)])
        else:
            for a0,a1 in itertools.product(range(p),aa):
                r = constants[a0]*constants[a1]*V[a0,a1]*S1[a0,a1]
                U[a0,a1] = r
                U[a1,a0] = r
        loading_bar(a2+1,shots,scaling=lambda x : x**(3/2))
    return X

# if the partition has no overlapping sets and we never update the variance graph,
#     we can speed up the allocation of measurements considerably
def non_overlapping_bayes_min_var(P,constants,psi,shots,aaa):
    # Inputs:
    #     P          - (pauli)           - Paulis in Hamiltonian
    #     constants  - (list{int})       - coefficients in Hamiltonian
    #     psi        - (numpy.array)     - ground state of Hamiltonian
    #     shots      - (int)             - number of samples to take
    #     aaa        - (list{list{int}}) - partition of Hamiltonian
    #     update_var - (int)             - number of turns between updating variance estimates
    # Outputs:
    #     (numpy.array{Dict}) - array of measurement outcome counts
    p,q = P.paulis(),P.qubits()
    X = np.array([[{(1,1):0,(1,-1):0,(-1,1):0,(-1,-1):0} for a1 in range(p)] for a0 in range(p)])
    A = bayes_variance_graph(X,constants)
    aaa = list(aaa)
    bb = [0 for aa in aaa]
    cc = [A.adj[aa][:,aa].sum() for aa in aaa]
    l = len(bb)
    for _ in range(shots):
        a2 = max(range(l),key=lambda x : cc[x])
        bb[a2] += 1
        cc[a2] /= (bb[a2]*(bb[a2]+1))
    loading_bar(0,shots)
    for a2 in range(len(aaa)):
        aa = aaa[a2]
        P1 = P.copy()
        P1.delete_paulis_([a1 for a1 in range(P.paulis()) if not a1 in aa])
        C = diagonalize(P1)
        neg = negations(P1,C)
        P1 = act(P1,C)
        psi_diag = C.unitary() @ psi
        cdf = [0]
        for a3 in range(len(psi_diag)):
            cdf.append(cdf[-1]+np.absolute(psi_diag[a3])**2)
        l = len(cdf)
        p1 = len(aa)
        for j in range(bb[a2]):
            r = random.uniform(0,cdf[-1])
            sample = np.array([[max(np.where([c<=r for c in cdf])[0])&(l>>(a3+1)) for a3 in range(q)]],dtype=bool)
            mm = [(-1)**(neg[a]^functools.reduce(lambda x,y:x^y,(sample&P1.Z)[a,:])) for a in range(p1)]
            for (a0,m0),(a1,m1) in itertools.product(zip(aa,mm),repeat=2):
                X[a0,a1][(m0,m1)] += 1
            loading_bar(sum(bb[:a2])+j+1,shots)
    return X







