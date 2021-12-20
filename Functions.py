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
from functools import reduce
from operator import mul



# NAMING CONVENTIONS

# a - int
# aa - list{int}
# aaa - list{list{int}}
# b - bool
# c - float (constant)
# r - random
# i - indexing
# ss - str
# m - matrix (numpy.array or scipy.sparse.csr_matrix)
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
    sss0 = f.readlines()
    f.close()
    sss1 = []
    cc = []
    for i in range(0,len(sss0),2):
        sss1.append(sss0[i][0:-1])
        cc.append(float(sss0[i+1][1:-5]))
    return string_to_pauli(sss1),cc

# write the ground state to a file for easy lookup
def write_ground_state(path,psi):
    # Inputs:
    #     path - (str)         - path to desired ground state file
    #     psi  - (numpy.array) - ground state of Hamiltonian
    ss = list(psi[i] for i in range(len(psi)))
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
    sss = f.readlines()
    f.close()
    psi = []
    for ss in sss:
        psi.append([np.complex(ss1) for ss1 in ss.split(" ")])
    return np.array(*psi)

# write the true variance graph to a file for easy lookup
def write_variance_graph(path,A):
    # Inputs:
    #     path - (str)   - path to desired variance graph file
    #     A    - (graph) - variance graph of Hamiltonian
    sss = list(list(str(A.adj[i0,i1]) for i1 in range(A.ord())) for i0 in range(A.ord()))
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
    sss = f.readlines()
    f.close()
    m = []
    for ss in sss:
        m.append([float(ss1) for ss1 in ss.split(" ")])
    return graph(np.array(m))

# call within a loop to produce a loading bar which can be scaled for various timings
def loading_bar(runs,length=50,scalings=[]):
    # Inputs:
    #     runs    - (list{tuple{int}}) - list of pairs: current iteration, total iterations
    #     length  - (int)              - length of bar
    #     scaling - (list{function})   - scale speed of progress
    a0 = len(runs)
    scalings += [lambda x:x]*(a0-len(scalings))
    a1 = scalings[0](runs[0][0])+sum(scalings[i](runs[i][0])*scalings[i-1](runs[i-1][1]) for i in range(1,a0))
    a2 = reduce(mul,[scalings[i](runs[i][1]) for i in range(a0)])
    # for scale in scalings[::-1]:
    #     a1,a2 = scale(a1),scale(a2)
    ss0 = ("{0:.1f}").format(int(1000*a1/a2)/10)
    ss1 = ' '.join(str(runs[i][0])+'/'+str(runs[i][1]) for i in range(a0))
    bar = '█'*int(length*a1/a2) + '-'*(length-int(length*a1/a2))
    print(f' |{bar}| {ss0}% {ss1}', end="\r")
    if runs[0][0] >= runs[0][1]-1: 
        print(" "*(length+6+len(ss0)+len(ss1)),end="\r")



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
        return not np.any(self.Z)

    # check whether self has only Z component 
    def is_IZ(self):
        # Outputs:
        #     (bool) - True if self has only Z componenet, False otherwise
        return not np.any(self.X)

    # check whether the set of Paulis are pairwise commuting
    def is_commuting(self):
        # Outputs:
        #     (bool) - True if self is pairwise commuting set of Paulis
        p = self.paulis()
        PP = [self.a_pauli(i) for i in range(p)]
        return not any(symplectic_inner_product(PP[i0],PP[i1]) for i0,i1 in itertools.combinations(range(p),2))

    # check whether the set of Paulis are pairwise commuting on every qubit
    def is_qubitwise_commuting(self):
        # Outputs:
        #     (bool) - True if self is pairwise qubitwise commuting set of Paulis
        p = self.paulis()
        PP = [self.a_pauli(i) for i in range(p)]
        return not any(any((PP[i0].X[0,i2]&PP[i1].Z[0,i2])^(PP[i0].Z[0,i2]&PP[i1].X[0,i2]) for i2 in range(self.qubits())) for i0,i1 in itertools.combinations(range(p),2))

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
        return self

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
        X = np.array([[self.X[i0,i1] for i1 in range(self.qubits())] for i0 in range(self.paulis())],dtype=bool)
        Z = np.array([[self.Z[i0,i1] for i1 in range(self.qubits())] for i0 in range(self.paulis())],dtype=bool)
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
        for i in range(self.paulis()):
            print(''.join(str(int(i1)) for i1 in self.X[i,:]),''.join(str(int(i1)) for i1 in self.Z[i,:]))

# convert a pauli object to its matrix representation
def pauli_to_matrix(P):
    # Inputs:
    #     P - (pauli) - must have shape (1,q)
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of input Pauli
    if P.paulis() != 1:
        raise Exception("Matrix can only be constructed for a single Pauli")
    X,Z = P.X[0],P.Z[0]
    mDict = {(0,0):I_mat,(0,1):Z_mat,(1,0):X_mat,(1,1):Y_mat}
    return tensor([mDict[(X[i],Z[i])] for i in range(P.qubits())])

# convert a collection of strings (or single string) to a pauli object
def string_to_pauli(sss):
    # Inputs:
    #     sss - (list{str}) or (str) - string representation of Pauli
    # Outputs:
    #     (pauli) - Pauli corresponding to input string(s)
    XDict = {"I":0,"X":1,"Y":1,"Z":0}
    ZDict = {"I":0,"X":0,"Y":1,"Z":1}
    if type(sss) is str:
        X = np.array([[XDict[s] for s in sss]],dtype=bool)
        Z = np.array([[ZDict[s] for s in sss]],dtype=bool)
        return pauli(X,Z)
    else:
        X = np.array([[XDict[s] for s in ss] for ss in sss],dtype=bool)
        Z = np.array([[ZDict[s] for s in ss] for ss in sss],dtype=bool)
        return pauli(X,Z)

# convert a pauli object to a collection of strings (or single string)
def pauli_to_string(P):
    # Inputs:
    #     P - (pauli) - Pauli to be stringified
    # Outputs:
    #     (list{str}) - string representation of Pauli
    X,Z = P.X,P.Z
    ssDict = {(0,0):"I",(0,1):"Z",(1,0):"X",(1,1):"Y"}
    if P.paulis() == 0:
        return ''
    elif P.paulis() == 1:
        return ''.join(ssDict[(X[0,i],Z[0,i])] for i in range(P.qubits()))
    else:
        return [''.join(ssDict[(X[i0,i1],Z[i0,i1])] for i1 in range(P.qubits())) for i0 in range(P.paulis())]

# the symplectic inner product of two pauli objects (each with a single Pauli)
def symplectic_inner_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (int) - symplectic inner product of Paulis modulo 2
    if (P0.paulis() != 1) or (P1.paulis() != 1):
        raise Exception("Symplectic inner product only works with pair of single Paulis")
    if P0.qubits() != P1.qubits():
        raise Exception("Symplectic inner product only works if Paulis have same number of qubits")
    return functools.reduce(lambda x,y:x^y,np.logical_xor(np.logical_and(P0.X,P1.Z),np.logical_and(P0.Z,P1.X))[0,:])

# the symplectic inner product of two pauli objects (each with a single Pauli)
def qubitwise_inner_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (int) - qubitwise inner product of Paulis modulo 2
    if (P0.paulis() != 1) or (P1.paulis() != 1):
        raise Exception("Qubitwise inner product only works with pair of single Paulis")
    if P0.qubits() != P1.qubits():
        raise Exception("Qubitwise inner product only works if Paulis have same number of qubits")
    return any((P0.X[0,i]&P1.Z[0,i])^(P0.Z[0,i]&P1.X[0,i]) for i in range(P0.qubits()))

# the product of two pauli objects
def pauli_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (pauli) - product of Paulis
    if P0.paulis() != 1 or P1.paulis() != 1:
        raise Exception("Product can only be calculated for single Paulis")
    return pauli(np.logical_xor(P0.X,P1.X),np.logical_xor(P0.Z,P1.Z))



# GATES & CIRCUITS

# a class for storing quantum gates as a name and a list of qubits
class gate:
    def __init__(self,name,aa):
        # Inputs:
        #     name - (function)  - function for action of gate
        #     aa   - (list{int}) - list of qubits acted upon
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
            self.gg = [self.gg[i] for i in range(self.length()) if not i in aa]

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
    return tensor([H_mat if i in aa else I_mat for i in range(q)])

# function for mapping phase gate to corresponding unitary matrix
def S_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - indices for phase gate tensors
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional S(aa)
    return tensor([S_mat if i in aa else I_mat for i in range(q)])

# function for mapping CNOT gate to corresponding unitary matrix
def CX_unitary(aa,q):
    # Inputs:
    #     aa - (list{int}) - control aa[0] and target aa[1]
    #     q  - (int)       - number of qubits
    # Outputs:
    #     (scipy.sparse.csr_matrix) - matrix representation of q-dimensional CNOT(aa[0],aa[1])
    a0 = q-1-aa[0]
    a1 = q-1-aa[1]
    aa2 = np.array([1 for i in range(1<<q)])
    aa3 = np.array([i for i in range(1<<q)])
    aa4 = np.array([(i^((i&(1<<(a0)))>>a0)*(1<<(a1))) for i in range(1<<q)])
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
    aa2 = np.array([(-1)**(((i&(1<<(a0)))>>a0)&((i&(1<<(a1)))>>a1)) for i in range(1<<q)])
    aa3 = np.array([i for i in range(1<<q)])
    aa4 = np.array([i for i in range(1<<q)])
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
    aa2 = np.array([1 for i in range(1<<q)])
    aa3 = np.array([i for i in range(1<<q)])
    aa4 = np.array([i^(1<<a0)^(1<<a1) if (((i&(1<<a0))>>a0)^((i&(1<<a1))>>a1)) else i for i in range(1<<q)])
    return scipy.sparse.csr_matrix((aa2,(aa3,aa4)))

# returns the circuit which diagonalizes a pairwise commuting pauli object
def diagonalize(P):
    # Inputs:
    #     P - (pauli) - Pauli to be diagonalized
    # Outputs:
    #     (circuit) - circuit which diagonalizes P
    q = P.qubits()

    if not P.is_commuting():
        raise Exception("Paulis must be pairwise commuting to be diagonalized")
    P1 = P.copy()
    C = circuit(q)

    # for each qubit, call diagonalize_iter_
    for i in range(q):
        C = diagonalize_iter_(P1,C,i)
    P1 = act(P1,C)

    # if any qubits are X rather than Z, apply H to make them Z
    if [i for i in range(P1.qubits()) if any(P1.X[:,i])]:
        C.add_gates_(gate(H,[i for i in range(q) if any(P1.X[:,i])]))
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
    a1 = min(i for i in range(p) if P.X[i,a])

    # add CNOT gates to cancel out all non-zero X-parts on Pauli a1, qubits > a
    if any(P.X[a1,i] for i in range(a+1,q)):
        gg = [gate(CX,[a,i]) for i in range(a+1,q) if P.X[a1,i]]
        C.add_gates_(gg)
        P = act(P,gg)

    # check whether there are any non-zero Z-parts on Pauli a1, qubits > a
    if any(P.Z[a1,i] for i in range(a+1,q)):

        # if Pauli a1, qubit a is X, apply S gate to make it Y
        if not P.Z[a1,a]:
            g = gate(S,[a])
            C.add_gates_(g)
            P = act(P,g)

        # add backwards CNOT gates to cancel out all non-zero Z-parts on Pauli a1, qubits > a
        gg = [gate(CX,[i,a]) for i in range(a+1,q) if P.Z[a1,i]]
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
    def add_vertex_(self,c=1):
        # Inputs:
        #     c - (float) - vertex weight
        if len(self.adj) == 0:
            self.adj = np.array([c])
        else:
            m0 = np.zeros((len(self.adj),1))
            m1 = np.zeros((1,len(self.adj)))
            m2 = np.array([[c]])
            self.adj = np.block([[self.adj,m0],[m1,m2]])

    # weight a vertex
    def lade_vertex_(self,a,c):
        # Inputs:
        #     a - (int)   - vertex to be weighted
        #     c - (float) - vertex weight
        self.adj[a,a] = c

    # weight an edge
    def lade_edge_(self,a0,a1,c):
        # Inputs:
        #     a0 - (int)   - first vertex
        #     a1 - (int)   - second vertex
        #     c  - (float) - vertex weight
        self.adj[a0,a1] = c
        self.adj[a1,a0] = c

    # returns a set of the neighbors of a given vertex
    def neighbors(self,a):
        # Inputs:
        #     a - (int) - vertex for which neighbors should be returned
        # Outputs:
        #     (list{int}) - set of neighbors of vertex a
        aa1 = set([])
        for i in range(self.ord()):
            if (a != i) and (self.adj[a,i] != 0):
                aa1.add(i)
        return aa1

    # returns list of all edges in self
    def edges(self):
        # Outputs:
        #     (list{list{int}}) - list of edges in self
        aaa = []
        for i0,i1 in itertools.combinations(range(self.ord()),2):
            if i1 in self.neighbors(i0):
                aaa.append([i0,i1])
        return aaa

    # check whether a collection of vertices is a clique in self
    def clique(self,aa):
        # Inputs:
        #     aa - (list{int}) - list of vertices to be checked for clique
        # Outputs:
        #     (bool) - True if aa is a clique in self; False otherwise
        for i0,i1 in itertools.combinations(aa,2):
            if self.adj[i0,i1] == 0:
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
        for i0 in range(self.ord()):
            print('[',end=' ')
            for i1 in range(self.ord()):
                s = self.adj[i0,i1]
                if str(s)[0] == '-':
                    print(f'{self.adj[i0,i1]:.2f}',end=" ")
                else:
                    print(' '+f'{self.adj[i0,i1]:.2f}',end=" ")
            print(']')

    # print self as a list of vertices together with their neighbors
    def print_neighbors(self):
        for i0 in range(self.ord()):
            print(i0,end=": ")
            for i1 in self.neighbors(i0):
                print(i1,end=" ")
            print()

    # return a deep copy of self
    def copy(self):
        # Outputs:
        #     (graph) - deep copy of self
        return graph(np.array([[self.adj[i0,i1] for i1 in range(self.ord())] for i0 in range(self.ord)]))

# returns all non-empty cliques in a graph
def nonempty_cliques(A):
    # Inputs:
    #     A - (graph) - graph for which all cliques should be found
    # Outputs:
    #     (list{list{int}}) - a list containing all non-empty cliques in A
    p = A.ord()
    aaa = set([frozenset([])])
    for i in range(p):
        iset = set([i])
        inter = A.neighbors(i)
        aaa |= set([frozenset(iset|(inter&aa)) for aa in aaa])
    aaa.remove(frozenset([]))
    return list([list(aa) for aa in aaa])

# returns an generator over all maximal cliques in a graph
def all_maximal_cliques(A):
    # Inputs:
    #     A - (graph) - graph for which all cliques should be found
    # Outputs:
    #     (generator) - a generator over all maximal cliques in A
    p = A.ord()
    N = {}
    for i in range(p):
        N[i] = A.neighbors(i)
    nxG = nx.Graph()
    nxG.add_nodes_from([i for i in range(p)])
    nxG.add_edges_from([(i0,i1) for i0 in range(p) for i1 in N[i0]])
    return nx.algorithms.clique.find_cliques(nxG)

# returns a clique covering of a graph which hits every vertex at least a certain number of times
def weighted_vertex_covering_maximal_cliques(A,A1=None,cc=None,k=1):
    # Inputs:
    #     A  - (graph)     - commutation graph for which covering should be found
    #     A1 - (graph)     - variance graph for which covering should be found
    #     cc - (list{int}) - coefficients of the Hamiltonian
    #     k  - (int)       - number of times each vertex should be covered
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which cover A
    p = A.ord()
    if A1 == None and cc == None:
        return vertex_covering_maximal_cliques(A,k=k)
    elif A1 == None:
        cc2 = [cc[i]**2 for i in range(p)]
        N = {}
        for i in range(p):
            N[i] = A.neighbors(i)
        aaa = []
        for i0 in range(p):
            for i1 in range(k):
                aa0 = [i0]
                aa1 = list(N[i0])
                while aa1:
                    c1 = sum(cc2[a0] for a0 in aa0)
                    cc1 = [c1+sum(cc2[a2] for a2 in N[a1].intersection(aa1)) for a1 in aa1]
                    if sum(cc1) == 0:
                        cc1 = [1 for a in aa1]
                    r = random.choices(aa1,cc1)[0]
                    aa0.append(r)
                    aa1 = list(N[r].intersection(aa1))
                aaa.append(aa0)
        return [sorted(list(aa1)) for aa1 in set([frozenset(aa) for aa in aaa])]
    else:
        V1 = A1.adj
        N = {}
        for i in range(p):
            N[i] = A.neighbors(i)
        N2 = {}
        for i in range(p):
            N2[i] = A.neighbors(i)|set([i])
        aaa = []
        for i0 in range(p):
            for i1 in range(k):
                aa0 = [i0]
                aa1 = list(N[i0])
                aa2 = aa0+aa1
                while aa1:
                    cc1 = [V1[list(N2[a1].intersection(aa2))][:,list(N2[a1].intersection(aa2))].sum() for a1 in aa1]
                    if sum(cc1) == 0:
                        cc1 = [1 for a in aa1]
                    r = random.choices(aa1,cc1)[0]
                    aa0.append(r)
                    aa1 = list(N[r].intersection(aa1))
                    aa2 = aa0+aa1
                aaa.append(aa0)
        return [sorted(list(aa1)) for aa1 in set([frozenset(aa) for aa in aaa])]

# returns a clique covering of a graph which hits every vertex at least a certain number of times
def vertex_covering_maximal_cliques(A,k=1):
    # Inputs:
    #     A - (graph) - commutation graph for which covering should be found
    #     k - (int)   - number of times each vertex must be covered
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which cover A
    p = A.ord()
    N = {}
    for i in range(p):
        N[i] = A.neighbors(i)
    aaa = []
    for i0 in range(p):
        for i1 in range(k):
            aa0 = [i0]
            aa1 = list(N[i0])
            while aa1:
                cc = [len(N[a1].intersection(aa1)) for a1 in aa1]
                if sum(cc) == 0:
                    cc = [1 for a in aa1]
                r = random.choices(aa1,cc)[0]
                aa0.append(r)
                aa1 = list(N[r].intersection(aa1))
            aaa.append(aa0)
    return [sorted(list(aa1)) for aa1 in set([frozenset(aa) for aa in aaa])]

# reduces a clique covering of a graph by removing cliques with lowest weight
def post_process_cliques(A,aaa,k=1):
    # Inputs:
    #     A   - (graph)           - varaince graph from which weights of cliques can be obtained
    #     aaa - (list{list{int}}) - a clique covering of the Hamiltonian
    #     k   - (int)             - number of times each vertex must be covered
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which cover A
    p = A.ord()
    V = A.adj
    s = np.array([sum([i in aa for aa in aaa]) for i in range(p)])
    D = {}
    for aa in aaa:
        D[str(aa)] = V[aa][:,aa].sum()
    aaa1 = aaa.copy()
    aaa1 = list(filter(lambda x : all(a>=(k+1) for a in s[aa]),aaa1))
    while aaa1:
        aa = min(aaa1,key=lambda x : D[str(x)])
        aaa.remove(aa)
        aaa1.remove(aa)
        s -= np.array([int(i in aa) for i in range(p)])
        aaa1 = list(filter(lambda x : all(a>=(k+1) for a in s[aa]),aaa1))
    return aaa

# returns a largest-degree-first clique partition of a graph
def LDF(A):
    # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    p = A.ord()
    remaining = set(range(p))
    N = {}
    for i in range(p):
        N[i] = A.neighbors(i)
    aaa = []
    while remaining:
        a = max(remaining,key=lambda x : len(N[x]&remaining))
        aa0 = set([a])
        aa1 = N[a]&remaining
        while aa1:
            a2 = max(aa1,key=lambda x : len(N[x]&aa1))
            aa0.add(a2)
            aa1 &= N[a2]
        aaa.append(aa0)
        remaining -= aa0
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

# returns the mean of a Hamiltonian with a given state
def Hamiltonian_Mean(P,cc,psi):
    # Inputs:
    #     P   - (pauli)       - Paulis of Hamiltonian
    #     cc  - (list{float}) - coefficients of Hamiltonian
    #     psi - (numpy.array) - state for mean
    # Outputs:
    #     (numpy.float64) - mean sum(c*<psi|P|psi>)
    p = P.paulis()
    return sum(cc[i]*Mean(P.a_pauli(i),psi) for i in range(p))

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
def Cov(P0,P1,psi):
    # Inputs:
    #     P0  - (pauli)       - first Pauli for covariance
    #     P1  - (pauli)       - second Pauli for covariance
    #     psi - (numpy.array) - state for covariance
    # Outputs:
    #     (numpy.float64) - covariance <psi|P0P1|psi> - <psi|P0|psi><psi|P1|psi>
    m0 = pauli_to_matrix(P0)
    m1 = pauli_to_matrix(P1)
    psi_dag = psi.conj().T
    cov = (psi_dag @ m0 @ m1 @ psi)-(psi_dag @ m0 @ psi)*(psi_dag @ m1 @ psi)
    return cov.real

# returns the graph of variances and covariances for a given Hamiltonian and ground state
def variance_graph(P,cc,psi):
    # Inputs:
    #     P   - (pauli)         - set of Paulis in Hamiltonian
    #     cc  - (list{float64}) - coefficients in Hamiltonian
    #     psi - (numpy.array)   - ground state
    # Outputs:
    #     (graph) - variances and covariances of all Paulis with respect to ground state
    p = P.paulis()
    mm = [pauli_to_matrix(P.a_pauli(i)) for i in range(p)]
    psi_dag = psi.conj().T
    cc1 = [(psi_dag@mm[i]@psi).real for i in range(p)]
    return graph(np.array([[cc[i0]*cc[i1]*((psi_dag@mm[i0]@mm[i1]@psi)-cc1[i0]*cc1[i1]).real for i1 in range(p)] for i0 in range(p)]))

# scales the entries in a variance graph with respect to number of measurements
def scale_variances(A,S):
    # Inputs:
    #     A - (graph)       - variance matrix
    #     S - (numpy.array) - array for tracking number of measurements
    p = A.ord()
    S1 = S.copy()
    S1[range(p),range(p)] = [a if a != 0 else 1 for a in S1.diagonal()]
    s1 = 1/S1.diagonal()
    return graph(S1*A.adj*s1*s1[:,None])

# returns the commutation graph of a given Pauli
def commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for commutation relations
    # Outputs:
    #     (graph) - an edge is weighted 1 if the pair of Paulis commute
    p = P.paulis()
    return graph(np.array([[1-symplectic_inner_product(P.a_pauli(i0),P.a_pauli(i1)) for i1 in range(p)] for i0 in range(p)]))

# returns the qubitwise commutation graph of a given Pauli
def qubitwise_commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for qubitwise commutation relations
    # Outputs:
    #     (graph) - an edge is weighted 1 if the pair of Paulis qubitwise commute
    p = P.paulis()
    return graph(np.array([[1-qubitwise_inner_product(P.a_pauli(i0),P.a_pauli(i1)) for i1 in range(p)] for i0 in range(p)]))

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
    for i in range(p):
        rr = random.sample(range(q),d)
        sss.append("".join([ssdict[random.randint(0,3)] if i1 in rr else "I" for i1 in range(q)]))
    return string_to_pauli(sss)

# print list of Paulis in string form, together with coefficients
def print_Ham_string(P,cc):
    # Inputs:
    #     P  - (pauli)     - Pauli to be printed
    #     cc - (list{int}) - coefficients for Hamiltonian
    X,Z = P.X,P.Z
    for i in range(P.paulis()):
        print(pauli_to_string(P.a_pauli(i)),end="")
        if cc[i] >= 0:
            print(" +%s"%cc[i])
        else:
            print(" %s"%cc[i])

# returns the ground state of a given Hamiltonian
def ground_state(P,cc):
    # Inputs:
    #     P  - (pauli)     - Paulis for Hamiltonian
    #     cc - (list{int}) - coefficients for Hamiltonian
    # Outputs:
    #     (numpy.array) - eigenvector corresponding to lowest eigenvalue of Hamiltonian
    m = sum(pauli_to_matrix(P.a_pauli(i))*cc[i] for i in range(P.paulis()))
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
        return np.logical_xor.reduce([P.X[:,a]&P.Z[:,a] for a in g.aa])
    elif g.name == H:
        return np.logical_xor.reduce([P.X[:,a]&P.Z[:,a] for a in g.aa])
    elif g.name == CX:
        a0,a1 = g.aa
        return P.X[:,a0]&P.Z[:,a1]&(P.Z[:,a0]==P.X[:,a1])
    return [False]*P.paulis()

# sample from distribution given by ground state and eigenstates of clique
#     optionally input a dictionary, which will be updated to track speed up future samples
def sample_(P,psi,aa,D={}):
    # Inputs:
    #     P   - (pauli)       - Paulis for Hamiltonian
    #     psi - (numpy.array) - ground state of Hamiltonian
    #     aa  - (list{int})   - clique to be measured
    #     D   - (dict)        - dictionary for storing pdf and negations for future samples
    # Outputs:
    #     (list{int}) - ith entry is +1/-1 for measurement outcome on ith element of aa
    if str(aa) in D.keys():
        P1,pdf,bb = D[str(aa)]
    else:
        P1 = P.copy()
        P1.delete_paulis_([i for i in range(P.paulis()) if not i in aa])
        C = diagonalize(P1)
        psi_diag = C.unitary() @ psi
        pdf = np.absolute(psi_diag*psi_diag.conj())
        bb = negations(P1,C)
        P1 = act(P1,C)
        D[str(aa)] = (P1,pdf,bb)
    p,q = P1.paulis(),P1.qubits()
    a1 = np.random.choice(1<<q,p=pdf)
    s = np.array([[a1&(1<<(q-1-i)) for i in range(q)]],dtype=bool)&P1.Z
    return [(-1)**(bb[i]^functools.reduce(lambda x,y:x^y,s[i])) for i in range(p)]



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
def bayes_variance_graph(X,cc):
    # Inputs:
    #     X  - (numpy.array{dict}) - array for tracking measurement outcomes
    #     cc - (list{float})       - coefficients of Hamiltonian
    # Outputs:
    #     (numpy.array{float}) - variance graph calculated with Bayesian estimates
    p = len(cc)
    return graph(np.array([[(cc[i0]**2)*bayes_Var(X[i0,i0]) if i0==i1 else cc[i0]*cc[i1]*bayes_Cov(X[i0,i1],X[i0,i0],X[i1,i1]) for i1 in range(p)] for i0 in range(p)]))

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
def naive_variance_graph(X,cc):
    # Inputs:
    #     X  - (numpy.array{dict}) - array for tracking measurement outcomes
    #     cc - (list{float})       - coefficients of Hamiltonian
    # Outputs:
    #     (numpy.array{float}) - variance graph calculated with naive estimates
    p = len(cc)
    return graph(np.array([[(cc[i0]**2)*naive_Var(X[i0,i0]) if i0==i1 else cc[i0]*cc[i1]*naive_Cov(X[i0,i1],X[i0,i0],X[i1,i1]) for i1 in range(p)] for i0 in range(p)]))



# SIMULATION ALGORITHMS

# convert from L,l notation to set of update steps
def Ll_updates(L,l,shots):
    # Inputs:
    #     L     - (int) - number of sections into which shots should be split
    #     l     - (int) - exponential scaling factor for size of sections
    #     shots - (int) - total number of shots required
    # Outputs:
    #     (set{int}) - set containing steps at which algorithm should update
    r0_shots = shots/sum([(1+l)**i for i in range(L)])
    shot_nums = [round(r0_shots*(1+l)**i) for i in range(L-1)]
    shot_nums.append(shots-sum(shot_nums))
    return set([0]+list(itertools.accumulate(shot_nums))[:-1])

# updates the variance matrix by sampling from pre-determined cliques
def variance_estimate_(P,cc,psi,D,X,xxx):
    # Inputs:
    #     P   - (pauli)             - Paulis in Hamiltonian
    #     cc  - (list{int})         - coefficients in Hamiltonian
    #     psi - (numpy.array)       - ground state of Hamiltonian
    #     D   - (dict)              - dictionary for storing pdf and negations for future samples
    #     X   - (numpy.array{dict}) - array of measurement outcome counts
    #     xxx - (list{list{int}})   - list of cliques to-be-sampled
    # Outputs:
    #     (numpy.array{float}) - variance graph calculated with Bayesian estimates
    #     (dict)               - (updated) dictionary for storing pdf and negations for future samples
    #     (numpy.array{dict})  - (updated) array of measurement outcome counts
    p = P.paulis()
    index_set = set(range(p))
    for aa in xxx:
        aa1 = sorted(index_set.difference(aa))
        cc1 = sample_(P,psi,aa,D)
        for (a0,c0),(a1,c1) in itertools.product(zip(aa,cc1),repeat=2):
            X[a0,a1][(c0,c1)] += 1
    return bayes_variance_graph(X,cc).adj,D,X

# partitions Hamiltonian and repeatedly samples cliques while minimizing total variance
#     returns an array of dictionaries which tracks ++/+-/-+/-- outcomes for each pair of Paulis
def bucket_filling(P,cc,psi,shots,part_func,update_steps=set([]),repeats=(0,1),full_simulation=False):
    # Inputs:
    #     P               - (pauli)       - Paulis in Hamiltonian
    #     cc              - (list{int})   - coefficients in Hamiltonian
    #     psi             - (numpy.array) - ground state of Hamiltonian
    #     shots           - (int)         - number of samples to take
    #     part_func       - (function)    - function for determining partition
    #     update_steps    - (set{int})    - steps at which variance graph should be updated
    #     repeats         - (tuple{int})  - current iteration and total number of iterations
    #     full_simulation - (bool)        - set True if full simulation is required
    # Outputs:
    #     (numpy.array{int})  - array containing number of times each pair of Paulis was measured together
    #     (numpy.array{dict}) - array of measurement outcome counts
    #     (list{list{int}})   - list of cliques which were sampled
    if part_func == None:
        return non_partitioning_bayes_min_var(P,cc,psi,shots,update_steps=update_steps,full_simulation=full_simulation)
    p = P.paulis()
    X = np.array([[{(1,1):0,(1,-1):0,(-1,1):0,(-1,-1):0} for a1 in range(p)] for a0 in range(p)])
    if part_func == weighted_vertex_covering_maximal_cliques:
        aaa = part_func(commutation_graph(P),cc=cc,k=3)
        # aaa = post_process_cliques(bayes_variance_graph(X,cc),aaa,k=10)
    else:
        aaa = part_func(commutation_graph(P))
    aaa,aaa1,aaa2 = itertools.tee(aaa,3)
    if not update_steps&set(range(1,shots)) and not any(set(aa1)&set(aa2) for aa1,aa2 in itertools.product(aaa1,aaa2) if not aa1==aa2):
        return non_overlapping_bayes_min_var(P,cc,psi,shots,aaa,repeats=repeats)
    D = {}
    S = np.zeros((p,p),dtype=int)
    Ones = [np.ones((i,i),dtype=int) for i in range(p+1)]
    index_set = set(range(p))
    xxx = []
    xxx1 = []
    for i0 in range(shots):
        if i0 == 0 or i0 in update_steps:
            V,D,X = variance_estimate_(P,cc,psi,D,X,xxx1)
            xxx1 = []
        S1 = S+Ones[p]
        s = 1/(S.diagonal()|(S.diagonal()==0))
        s1 = 1/S1.diagonal()
        factor = p-np.count_nonzero(S.diagonal())
        S1[range(p),range(p)] = [a if a != 1 else -factor for a in S1.diagonal()]
        V1 = V*(S*s*s[:,None] - S1*s1*s1[:,None])
        V2 = 2*V*(S*s*s[:,None] - S*s*s1[:,None])
        aaa,aaa1 = itertools.tee(aaa,2)
        aa = sorted(max(aaa1,key=lambda xx : V1[xx][:,xx].sum()+V2[xx][:,list(index_set.difference(xx))].sum()))
        xxx.append(aa)
        xxx1.append(aa)
        S[np.ix_(aa,aa)] += Ones[len(aa)]
        loading_bar([(i0,shots),repeats],scalings=[lambda x:x**(3/2)])
    if full_simulation:
        for aa in xxx1:
            aa1 = sorted(index_set.difference(aa))
            cc1 = sample_(P,psi,aa,D)
            for (a0,c0),(a1,c1) in itertools.product(zip(aa,cc1),repeat=2):
                X[a0,a1][(c0,c1)] += 1
    else:
        X = None
    return S,X,xxx

# if the partition has no overlapping sets, we can speed up the allocation of measurements
#     returns an array of dictionaries which tracks ++/+-/-+/-- outcomes for each pair of Paulis
def non_overlapping_bayes_min_var(P,cc,psi,shots,aaa,repeats=(0,1),full_simulation=False):
    # Inputs:
    #     P               - (pauli)           - Paulis in Hamiltonian
    #     cc              - (list{int})       - coefficients in Hamiltonian
    #     psi             - (numpy.array)     - ground state of Hamiltonian
    #     shots           - (int)             - number of samples to take
    #     aaa             - (list{list{int}}) - partition of Hamiltonian
    #     repeats         - (tuple{int})      - current iteration and total number of iterations
    #     full_simulation - (bool)            - set True if full simulation is required
    # Outputs:
    #     (numpy.array{int})  - array containing number of times each pair of Paulis was measured together
    #     (numpy.array{dict}) - array of measurement outcome counts
    #     (list{list{int}})   - list of cliques which were sampled
    p,q = P.paulis(),P.qubits()
    X = np.array([[{(1,1):0,(1,-1):0,(-1,1):0,(-1,-1):0} for a1 in range(p)] for a0 in range(p)])
    S = np.zeros((p,p),dtype=int)
    Ones = [np.ones((i,i),dtype=int) for i in range(p+1)]
    A = bayes_variance_graph(X,cc)
    aaa = list(aaa)
    aa1 = [0 for aa in aaa]
    aa2 = [np.sqrt(A.adj[aa][:,aa].sum()) for aa in aaa]
    aa1_len = len(aa1)
    for i in range(shots):
        a3 = max(range(aa1_len),key=lambda x : aa2[x])
        aa1[a3] += 1
        aa2[a3] *= ((aa1[a3]-1) or not (aa1[a3]-1))/(aa1[a3]+1)
    xxx = [aaa[i0] for i0 in list(itertools.chain.from_iterable([[i2]*aa1[i2] for i2 in range(len(aaa))]))]
    for xx in xxx:
        S[np.ix_(xx,xx)] += Ones[len(xx)]
    l = 1<<(q-1)
    if full_simulation:
        for i in range(len(aaa)):
            aa = aaa[i]
            P1 = P.copy()
            P1.delete_paulis_([a1 for a1 in range(P.paulis()) if not a1 in aa])
            C = diagonalize(P1)
            neg = negations(P1,C)
            P1 = act(P1,C)
            psi_diag = C.unitary() @ psi
            pdf = np.absolute(psi_diag*psi_diag.conj())
            p1 = len(aa)
            for i1 in range(aa1[i]):
                a1 = np.random.choice(1<<q,p=pdf)
                s = np.array([[a1&(l>>a3) for a3 in range(q)]],dtype=bool)&P1.Z
                bb = [(-1)**(neg[a]^functools.reduce(lambda x,y:x^y,s[a])) for a in range(p1)]
                for (a0,b0),(a1,b1) in itertools.product(zip(aa,bb),repeat=2):
                    X[a0,a1][(b0,b1)] += 1
                loading_bar([(sum(aa1[:i])+i1,shots),repeats],scalings=[lambda x:x**(3/2)])
    else:
        X = None
    return S,X,xxx






