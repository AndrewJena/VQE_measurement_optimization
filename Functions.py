import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import itertools
import random
import time
import math
import functools
import re
import os
import networkx as nx
from scipy import ndimage
# import sys
# sys.setrecursionlimit(40)



# NAMING CONVENTIONS

# a,b,c are integers/indices
# aa,bb,cc are lists/sets of integers
# aaa,bbb,ccc are lists/sets of lists of integers
# s,t are the same for strings
# m,n are the same for matrices
# u,v are the same for vertices
# P are paulis
# G are gates
# A are graphs (i.e. adjacency matrices)
# p is number of paulis
# q is number of qubits



# DEFINITIONS

# I,X,Y,Z Pauli matrices used for constructing tensor products
I_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]],dtype=complex))
X_mat = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]],dtype=complex))
Y_mat = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]],dtype=complex))
Z_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]],dtype=complex))
H_mat = scipy.sparse.csr_matrix(1/np.sqrt(2)*np.array([[1,1],[1,-1]],dtype=complex))
S_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,1j]],dtype=complex))



# GENERAL FUNCTIONS

def tensor(mm):
    # Input: list of matrices
    # Output: tensor product of matrices
    if len(mm) == 0:
        return matrix([])
    elif len(mm) == 1:
        return mm[0]
    else:
        return scipy.sparse.kron(mm[0],tensor(mm[1:]),format="csr")



# PAULIS

class pauli:
    # Input: X,Z matrices over booleans (representing a list of Paulis)
        # Paulis are stored using symplectic formalism
    def __init__(self,X,Z):
        self.X = X
        self.Z = Z

    def is_IX(self):
        # Input: none
        # Output: True if Pauli has no Z components; False otherwise
        if np.any(self.Z):
            return False
        return True

    def is_IZ(self):
        # Input: none
        # Output: True if Pauli has no X components; False otherwise
        if np.any(self.X):
            return False
        return True

    def a_pauli(self,a):
        # Input: a Int
        # Output: pauli object consisting of a single Pauli
        return pauli(np.array([self.X[a,:]]),np.array([self.Z[a,:]]))

    def paulis(self):
        # Input: none
        # Output: number of paulis
        return self.X.shape[0]

    def qubits(self):
        # Input: none
        # Output: number of qubits
        return self.X.shape[1]

    def delete_paulis(self,aa):
        # Input: aa list of ints
        # Output: Pauli after deletion of aa
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=0)
            self.Z = np.delete(self.Z,aa,axis=0)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=0)
                self.Z = np.delete(self.Z,a,axis=0)

    def delete_qubits(self,aa):
        # Input: aa list of ints
        # Output: Pauli after deletion of qubits
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=1)
            self.Z = np.delete(self.Z,aa,axis=1)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=1)
                self.Z = np.delete(self.Z,a,axis=1)

    def X_weight(self):
        # Input: none
        # Output: number of nonzero X entries
        if self.paulis() != 1:
            raise Exception("X weight can only be determined for a single Pauli")
        return np.count_nonzero(self.X)

    def copy(self):
        X = np.array([[self.X[b,a] for a in range(self.qubits())] for b in range(self.paulis())],dtype=bool)
        Z = np.array([[self.Z[b,a] for a in range(self.qubits())] for b in range(self.paulis())],dtype=bool)
        return pauli(X,Z)

    def print(self):
        sss = pauli_to_string(self)
        if type(sss) is str:
            print(sss)
        else:
            for ss in sss:
                print(ss)

    def print_symplectic(self):
        for a in range(self.paulis()):
            print(''.join(str(int(b)) for b in self.X[a,:]),''.join(str(int(b)) for b in self.Z[a,:]))


def pauli_to_matrix(P):
    # Input: P pauli (a single Pauli rather than a list)
    # Output: matrix representing tensor product of Pauli matrices
    if P.paulis() != 1:
        raise Exception("Matrix can only be constructed for a single Pauli")
    X,Z = P.X[0],P.Z[0]
    mmdict = {(0,0):I_mat,(0,1):Z_mat,(1,0):X_mat,(1,1):Y_mat}
    return tensor([mmdict[(X[a],Z[a])] for a in range(P.qubits())])

def string_to_pauli(sss):
    # Input: sss list of strings
    # Output: pauli object
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

def pauli_to_string(P):
    # Input: P pauli
    # Output: list of pauli strings
    X,Z = P.X,P.Z
    ssdict = {(0,0):"I",(0,1):"Z",(1,0):"X",(1,1):"Y"}
    if P.paulis() == 0:
        return ''
    elif P.paulis() == 1:
        return ''.join(ssdict[(X[0,a],Z[0,a])] for a in range(P.qubits()))
    else:
        return [''.join(ssdict[(X[b,a],Z[b,a])] for a in range(P.qubits())) for b in range(P.paulis())]

def symplectic_inner_product(P1,P2):
    # Input: P1,P2 paulis (single Paulis rather than lists of Paulis)
    # Output: the symplectic inner product modulo 2
    if (P1.paulis() != 1) or (P2.paulis() != 1):
        raise Exception("Symplectic inner product only works with pair of single Paulis")
    if P1.qubits() != P2.qubits():
        raise Exception("Symplectic inner product only works if Paulis have same number of qubits")
    X1,Z1,X2,Z2 = P1.X[0],P1.Z[0],P2.X[0],P2.Z[0]
    ip = 0
    for a in range(P1.qubits()):
        if X1[a] and Z2[a]:
            ip += 1
        if Z1[a] and X2[a]:
            ip += 1
    return ip%2

def restrict_to_paulis(P,aa):
    # Input: aa list of ints
    # Output: Pauli after restriction to aa
    Q = P.copy()
    for a in sorted([b for b in range(Q.paulis()) if not b in aa],reverse=True):
        Q.X = np.delete(Q.X,a,axis=0)
        Q.Z = np.delete(Q.Z,a,axis=0)
    return Q



# GATES & CIRCUITS

class gate:
    # Input: name function; qubits list of targets
    def __init__(self,name,aa):
        self.name = name
        self.aa = aa

    def name_string(self):
        return self.name.__name__

    def copy(self):
        return gate(self.name,[a for a in self.aa])

    def print(self):
        print("%s(%s)"%(self.name_string(),str(self.aa).replace(' ','')[1:-1]))

class circuit:
    # A list of gate (or Pauli) instances on self.dim qubits with an ordering given by self.gg
    def __init__(self,dim):
        # Initialize with the number of qubits, dim, and possibly a list of gates
        self.dim = dim
        self.gg = []

    def length(self):
        # Return the number of gate instances in the circuit (i.e. the number of time-steps)
        return len(self.gg)

    def unitary(self):
        U = scipy.sparse.csr_matrix(([1]*(1<<self.dim),(range(1<<self.dim),range(1<<self.dim))))
        for g in self.gg:
            U = globals()[g.name_string()+'_unitary'](g.aa,self.dim) @ U
        return U

    def add_gates(self,gg):
        # Add a list of gates, gg, to the end of the circuit (in order given by the list)
        if type(gg) is gate:
            self.gg.append(gg)
        elif type(gg) is circuit:
            self.gg += gg.gg
        else:
            self.gg += gg

    def insert_gates(self,gg,a):
        # Insert a gate, g, into a specific time-step, a, in the circuit
        if type(gg) is gate:
            self.gg.insert(a,gg)
        elif type(gg) is circuit:
            self.gg[a:a] = gg.gg
        else:
            self.gg[a:a] = gg

    def delete_gates(self,aa):
        # Delete the gate at specific time-steps, aa, in the circuit
        if type(aa) is int:
            del self.gg[aa]
        else:
            self.gg = [self.gg[b] for b in range(self.length()) if not b in aa]

    def copy(self):
        # A deep copy of a circuit instance
        return circuit(self.dim,[g.copy() for g in self.gg])

    def print(self):
        for g in self.gg:
            g.print()


def act(P,C):
    # Input: P pauli; C gate, circuit, or list of gates
    # Output: result of C acting on P (by conjugation)
    if type(C) is gate:
        return C.name(P,C.aa)
    elif type(C) is circuit:
        return act(P,C.gg)
    elif len(C) == 0:
        return P
    elif len(C) == 1:
        return act(P,C[0])
    else:
        return act(act(P,C[0]),C[1:])

def H(P,aa):
    # Input: P pauli; aa list of ints (target,)
    # Output: result of Hadamard acting on P on qubit aa[0]
        # X -> Z
        # Z -> X
    X,Z = P.X,P.Z
    for a in aa:
        X[:,a],Z[:,a] = Z[:,a].copy(),X[:,a].copy()
    return pauli(X,Z)

def S(P,aa):
    # Input: P pauli; aa list of ints (target,)
    # Output: result of S gate (phase gate) acting on P on qubit aa[0]
        # X -> Y
        # Z -> Z
    X,Z = P.X,P.Z
    for a in aa:
        Z[:,a] ^= X[:,a]
    return pauli(X,Z)

def CX(P,aa):
    # Input: P pauli; aa list of ints (control,target)
    # Output: result of CNOT acting on P with control aa[0] and target aa[1]
        # XI -> XX
        # IX -> IX
        # ZI -> ZI
        # IZ -> ZZ
    X,Z = P.X,P.Z
    a0,a1 = aa[0],aa[1]
    X[:,a1] ^= X[:,a0]
    Z[:,a0] ^= Z[:,a1]
    return pauli(X,Z)

def CZ(P,aa):
    # Input: P pauli; aa list of ints (target,target)
    # Output: result of CZ acting on P on qubits aa[0],aa[1]
        # XI -> XZ
        # IX -> ZX
        # ZI -> ZI
        # IZ -> IZ
    X,Z = P.X,P.Z
    a0,a1 = aa[0],aa[1]
    Z[:,a0] ^= X[:,a1]
    Z[:,a1] ^= X[:,a0]
    return pauli(X,Z)

def SWAP(P,aa):
    # Input: P pauli; aa list of ints (target,target)
    # Output: result of SWAP acting on P on qubits aa[0],aa[1]
        # XI -> IX
        # IX -> XI
        # ZI -> IZ
        # IZ -> ZI
    X,Z = P.X,P.Z
    a0,a1 = aa[0],aa[1]
    X[:,a0],X[:,a1] = X[:,a1].copy(),X[:,a0].copy()
    Z[:,a0],Z[:,a1] = Z[:,a1].copy(),Z[:,a0].copy()
    return pauli(X,Z)

def H_unitary(aa,dim):
    # Input: aa list of ints; dim int
    # Output: matrix which implements H on qubits a in aa
    return tensor([H_mat if a in aa else I_mat for a in range(dim)])

def S_unitary(aa,dim):
    # Input: aa list of ints; dim int
    # Output: matrix which implements S on qubits a in aa
    return tensor([S_mat if a in aa else I_mat for a in range(dim)])
        
def CX_unitary(aa,dim):
    # Input: aa list of two ints; dim int
    # Output: matrix which implements controlled-X with control aa[0] and target aa[1]
    b0 = dim-1-aa[0]
    b1 = dim-1-aa[1]
    bb1 = np.array([1 for b in range(1<<dim)])
    bb2 = np.array([b for b in range(1<<dim)])
    bb3 = np.array([(b^((b&(1<<(b0)))>>b0)*(1<<(b1))) for b in range(1<<dim)])
    return scipy.sparse.csr_matrix((bb1,(bb2,bb3)))

def CZ_unitary(aa,dim):
    # Input: aa list of two ints; dim int
    # Output: matrix which implements controlled-Z with controls aa[0] and aa[1]
    b0 = dim-1-aa[0]
    b1 = dim-1-aa[1]
    bb1 = np.array([(-1)**(((b&(1<<(b0)))>>b0)&((b&(1<<(b1)))>>b1)) for b in range(1<<dim)])
    bb2 = np.array([b for b in range(1<<dim)])
    bb3 = np.array([b for b in range(1<<dim)])
    return scipy.sparse.csr_matrix((bb1,(bb2,bb3)))

def SWAP_unitary(aa,dim):
    # Input: aa list of two ints; dim int
    # Output: matrix which implements SWAP with controls aa[0] and aa[1]
    b0 = dim-1-aa[0]
    b1 = dim-1-aa[1]
    bb1 = np.array([1 for b in range(1<<dim)])
    bb2 = np.array([b for b in range(1<<dim)])
    bb3 = np.array([b^(1<<b0)^(1<<b1) if (((b&(1<<b0))>>b0)^((b&(1<<b1))>>b1)) else b for b in range(1<<dim)])
    return scipy.sparse.csr_matrix((bb1,(bb2,bb3)))

def diagonalize(P):
    Q = P.copy()
    C = circuit(Q.qubits())
    for a in range(Q.qubits()):
        C = diagonalize_iter(Q,C,a)
    act(Q,C)
    if [a for a in range(Q.qubits()) if any(Q.X[:,a])]:
        C.add_gates(gate(H,[a for a in range(Q.qubits()) if any(Q.X[:,a])]))
    return C
 
def diagonalize_iter(P,C,a):
    Q = P.copy()
    act(Q,C)
    X,Z = Q.X,Q.Z
    if not any(X[:,a]):
        return C
    b = min(c for c in range(Q.paulis()) if X[c,a])

    if any(X[b,c] for c in range(a+1,Q.qubits())):
        C.add_gates([gate(CX,[a,c]) for c in range(a+1,Q.qubits()) if X[b,c]])
        act(Q,[gate(CX,[a,c]) for c in range(a+1,Q.qubits()) if X[b,c]])

    if any((c!=a)&(X[b,c]|Z[b,c]) for c in range(a+1,Q.qubits())):
        if not Z[b,a]:
            C.add_gates(gate(S,[a]))
            act(Q,gate(S,[a]))
        C.add_gates([gate(CX,[c,a]) for c in range(a+1,Q.qubits()) if (c!=a)&(X[b,c]|Z[b,c])])
        act(Q,[gate(CX,[c,a]) for c in range(a+1,Q.qubits()) if (c!=a)&(X[b,c]|Z[b,c])])

    if Z[b,a]:
        C.add_gates(gate(S,[a]))
        act(Q,gate(S,[a]))

    return C



# GRAPHS

class graph:
    # Input: order integer (number of vertices); adj_mat matrix (optional)
        # Graph objects store the adjacency matrix as self.adj
    def __init__(self,adj_mat=np.array([]),dtype=float):
        self.adj = adj_mat.astype(dtype)

    def add_vertex(self,weight):
        # Input: weight real
        # Output: graph with vertex added with proper weight (and no edges)
        if len(self.adj) == 0:
            self.adj = np.array([weight])
        else:
            r = np.zeros((1,len(self.adj)))
            c = np.zeros((len(self.adj),1))
            w = np.array([[weight]])
            self.adj = np.block([[self.adj,c],[r,w]])
        return self

    def lade_vertex(self,v,weight):
        # Input: v Int; weight real
        # Output: vertex v is given weight weight
        self.adj[v,v] = weight
        return self

    def lade_edge(self,v1,v2,weight):
        # Input: v1,v2 Ints; weight real
        # Output: edge uv (and vu since non-directed graph) is given weight weight
        self.adj[v1,v2] = weight
        self.adj[v2,v1] = weight
        return self

    def neighbors(self,v):
        # Input: v Int
        # Output: set of neighbors of vertex v
        ww = set([])
        for w in range(self.ord()):
            if (v != w) and (self.adj[v,w] != 0):
                ww.add(w)
        return ww

    def clique(self,vv):
        # Input: vv list of Ints
        # Output: True is vv is a clique in graph; false otherwise
        for v1,v2 in itertools.combinations(vv,2):
            if self.adj[v1,v2] == 0:
                return False
        return True

    def degree(self,v):
        # Input: v Int
        # Output: degree of vertex v
        return np.count_nonzero(self.adj[v,:])

    def ord(self):
        # Input: none
        # Output: number of vertices
        return self.adj.shape[0]

    def print(self):
        # Input: A graph
            # Print adjacency matrix of graph
        for a in range(self.ord()):
            print('[',end=' ')
            for b in range(self.ord()):
                s = self.adj[a,b]
                if str(s)[0] == '-':
                    print(f'{self.adj[a,b]:.4f}',end=" ")
                else:
                    print(' '+f'{self.adj[a,b]:.4f}',end=" ")
            print(']')

    def print_neighbors(self):
        # Input: A graph
            # Print vertex: neighbors view of graph
        for v in range(self.ord()):
            print(v,end=": ")
            for w in self.neighbors(v):
                print(w,end=" ")
            print()

    def copy(self):
        B = graph()
        for v in range(len(self.adj)):
            B.add_vertex(self.adj[v,v])
        for v1,v2 in zip(range(len(self.adj)),range(len(self.adj))):
            B.lade_edge(v1,v2,self.adj[v1,v2])
        return B


def greedy_min_parts(A):
    # Input: A graph
    # Output: list of lists of Ints (the partition into cliques)
        # Greedy algorithm for minimizing parts, not shots
    p = A.ord()
    aaa = []
    bb = [b for b in range(p)]
    random.shuffle(bb)
    for c in range(p):
        flag = False
        min_part = []
        for d in range(len(aaa)):
            aa = aaa[d]
            dd = aa+[bb[c]]
            if A.clique(dd):
                aaa[d] = dd
                flag = True
                break
        if not flag:
            aaa.append([bb[c]])
    return aaa

def remove_subsets(aaa):
    # remove subsets from a list of sets
    bbb = set([])
    for aa1,aa2 in itertools.combinations(aaa,2):
        if aa1 < aa2:
            bbb.add(aa1)
        elif aa2 < aa1:
            bbb.add(aa2)
    return aaa-bbb

def nonempty_cliques(A):
    # return all nonempty cliques of A
    p = A.ord()
    aaa = set([frozenset([])])
    for a in range(p):
        aset = set([a])
        inter = A.neighbors(a)
        aaa |= set([frozenset(aset|(inter&aa)) for aa in aaa])
    aaa.remove(frozenset([]))
    return list([list(aa) for aa in aaa])

def maximal_cliques(A):
    G = nx.Graph()
    G.add_nodes_from([a for a in range(A.ord())])
    G.add_edges_from([(a,b) for a in range(A.ord()) for b in A.neighbors(a)])
    return nx.algorithms.clique.find_cliques(G)

def covering_maximal_cliques(A,k):
    p = A.ord()
    aaa = []
    for b in range(p):
        for _ in range(k):
            cc = [b]
            while set.intersection(*[A.neighbors(c) for c in cc]):
                cc += random.sample(set.intersection(*[A.neighbors(c) for c in cc]),1)
            aaa.append(cc)
    return aaa





# PHYSICS FUNCTIONS

def Mean(P,psi):
    # Input: P pauli; psi np.array
    # Output: <psi|P|psi>
    m = pauli_to_matrix(P)
    psi_dag = psi.conj().T
    mean = psi_dag @ m @ psi
    return mean.real

def Var(P,psi):
    # Input: P pauli; psi np.array
    # Output: <psi|P^2|psi> - (<psi|P|psi>)^2
    m = pauli_to_matrix(P)
    psi_dag = psi.conj().T
    var = (psi_dag @ m @ m @ psi)-(psi_dag @ m @ psi)**2
    return var.real

def Cov(P1,P2,psi):
    # Input: P1,P2 paulis; psi np.array
    # Output: <psi|PQ|psi> - <psi|P|psi><psi|Q|psi>
    m1 = pauli_to_matrix(P1)
    m2 = pauli_to_matrix(P2)
    psi_dag = psi.conj().T
    cov = (psi_dag @ m1 @ m2 @ psi)-(psi_dag @ m1 @ psi)*(psi_dag @ m2 @ psi)
    return cov.real

def variance_graph(P,constants,psi):
    # Input: P pauli; constants list of reals; psi np.array
    # Output: graph representing variance and covariance of all Paulis w/r/t state psi
    n = P.paulis()
    A = graph()
    for a in range(n):
        Pa = P.a_pauli(a)
        A.add_vertex((constants[a]**2)*Var(Pa,psi).real)
    for a in range(n):
        Pa = P.a_pauli(a)
        for b in range(a+1,n):
            Pb = P.a_pauli(b)
            if not symplectic_inner_product(Pa,Pb):
                A.lade_edge(a,b,constants[a]*constants[b]*Cov(Pa,Pb,psi).real)
    return A

def commutation_graph(P):
    # Input: P pauli
    # Output: graph with edge if Paulis commute
    n = P.paulis()
    A = graph(dtype=bool)
    for a in range(n):
        A.add_vertex(1)
    for a in range(n):
        Pa = P.a_pauli(a)
        for b in range(a+1,n):
            Pb = P.a_pauli(b)
            if not symplectic_inner_product(Pa,Pb):
                A.lade_edge(a,b,1)
    return A

def anticommutation_graph(P):
    # Input: P pauli
    # Output: graph with edge if Paulis commute
    n = P.paulis()
    A = graph(dtype=bool)
    for a in range(n):
        A.add_vertex(1)
    for a in range(n):
        Pa = P.a_pauli(a)
        for b in range(a+1,n):
            Pb = P.a_pauli(b)
            if symplectic_inner_product(Pa,Pb):
                A.lade_edge(a,b,1)
    return A

def random_Ham(p,q,d):
    # Input: p,q,d ints (# Paulis, # qubits, max weight)
    # Output: random pauli satisfying input conditions
    sss = []
    ssdict = {0:"I",1:"Z",2:"X",3:"Y"}
    for i in range(p):
        sss.append("".join([ssdict[random.randint(0,3)] if j in random.sample(range(q),d) else "I" for j in range(q)]))
    return string_to_pauli(sss)

def print_Ham_string(P,constants):
    # Input: P pauli
        # Print list of Paulis in string form together with constants
    X,Z = P.X,P.Z
    for a in range(P.paulis()):
        print(pauli_to_string(P.a_pauli(a)),end="")
        if constants[a] >= 0:
            print(" +%s"%constants[a])
        else:
            print(" %s"%constants[a])

def ground_state(P,constants):
    # Input: P pauli
    # Output: eigenvector corresponding to lowest eigenvalue
    m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(P.paulis()))
    gval,gvec = scipy.sparse.linalg.eigsh(m,which='SA',k=1)
    return np.array([g for g in gvec[:,0]])




# MEASUREMENT FUNCTIONS

def neg(P,C):
    # Input: C circuit; dim int
    # Output: n bitmask
    Q = P.copy()
    n = 0
    for g in C.gg:
        n ^= neg_iter(Q,g)
        act(Q,g)
    return n

def neg_iter(P,g):
    # Input: P pauli; g gate
    # Output: bit
    if g.name == S:
        return functools.reduce(lambda i,j:i^j,[(P.X[0,a])&(P.Z[0,a]) for a in g.aa])
    elif g.name == H:
        return functools.reduce(lambda i,j:i^j,[(P.X[0,a])&(P.Z[0,a]) for a in g.aa])
    elif g.name == CX:
        a0,a1 = g.aa[0],g.aa[1]
        return (P.X[0,a0])&(P.Z[0,a1])&(P.Z[0,a0] == P.X[0,a1])
    return 0

def distribution(C,psi):
    # Input: C circuit; psi vector
    # Output: cdf for measurements
    psi_diag = C.unitary() @ psi
    cdf = [0]
    for b in range(len(psi_diag)):
        cdf.append(cdf[-1]+np.absolute(psi_diag[b])**2)
    return cdf

def sample_from_distribution(cdf):
    # Input: cdf list of floats
    # Output: int index sampled from distribution
    l = len(cdf)
    dim = int(math.log2(l))
    r = random.uniform(0,cdf[-1])
    s = 0
    a = max(np.where([c<=r for c in cdf])[0])
    return np.array([[a&(l>>(b+1)) for b in range(dim)]],dtype=bool)

def measurement_outcome(sample,P,C):
    # Input: sample bitarray; P Pauli; C Circuit
    # Output: measurement
    Q = P.copy()
    act(Q,C)
    if Q.paulis() != 1:
        raise Exception("Measurement outcome can only be found for a single Pauli")
    Z = Q.a_pauli(0).Z
    return (-1)**(neg(P,C)^functools.reduce(lambda i,j:i^j,(sample&Z)[0,:]))

def measure(P,psi,aa):
    Q = restrict_to_paulis(P,aa)
    C = diagonalize(Q)
    act(Q,C)
    cdf = distribution(C,psi)
    sample = sample_from_distribution(cdf)
    return [measurement_outcome(sample,P.a_pauli(aa[b]),C) for b in range(len(aa))]

def clique_to_measure(V,aaa):
    retmax = 0
    for aa in aaa:
        aamax = V[aa][:,aa].sum()
        if aamax > retmax:
            ret = aa
            retmax = aamax
    return ret

def greedy_bayes_min_var(P,constants,psi,aaa,shots):
    p = P.paulis()
    X = np.empty((p,p),dtype=object)
    for a,b in itertools.product(range(p),repeat=2):
        X[a,b] = ([],[])
    printProgressBar(progress_scaling(0),progress_scaling(shots))
    for b in range(shots):
        aaa,aaa1 = itertools.tee(aaa,2)
        aa = clique_to_measure(ready_to_measure_bvg(P,X,constants),aaa1)
        mm = measure(P,psi,aa)
        for (a0,m0),(a1,m1) in itertools.product(zip(aa,mm),repeat=2):
            X[a0,a1][0].append(m0)
            X[a0,a1][1].append(m1)
        printProgressBar(progress_scaling(b+1),progress_scaling(shots))
    return X

def scale_variances(A,X):
    p = A.ord()
    S0 = np.array([[len(X[a,b][0]) for b in range(p)] for a in range(p)],dtype=int)
    S1 = np.array([[S0[a,b]/(S0[a,a]*S0[b,b]) if S0[a,b]>0 else 0 for b in range(p)] for a in range(p)])
    B = A.copy()
    for a0,a1 in itertools.product(range(A.ord()),range(A.ord())):
        if S0[a0,a1] > 0:
            B.adj[a0,a1] *= S1[a0,a1]
    return B

def sample_Mean(xx):
    # return sample mean of outcomes xx
    if len(xx) == 0:
        return 0
    return sum(xx)/len(xx)

def sample_Var(xx,Mx):
    # return sample variance of outomces xx with mean Mxx
    if len(xx) == 0:
        return 1
    elif len(xx) == 1:
        return 0
    return sum((x-Mx)**2 for x in xx)/(len(xx)-1)

def sample_Cov(xx,yy,Mx,My):
    # return sample covariance of outcomes xx,yy with means Mxx,Myy
    if len(xx) <= 1:
        return 0
    return sum((x-Mx)*(y-My) for (x,y) in zip(xx,yy))/(len(xx)-1)

def sample_variance_graph(P,X,constants):
    # return sample variance graph
    p = P.paulis()
    MM = [sample_Mean(X[a,a][0]) for a in range(p)]
    S = np.array([[len(X[a,b][0]) for b in range(p)] for a in range(p)],dtype=int)
    V = np.zeros((p,p),dtype=float)
    for a in range(p):
        if S[a,a] > 0:
            V[a,a] = constants[a]*constants[a]*sample_Var(X[a,a][0],sample_Mean(X[a,a][0]))
        else:
            V[a,a] = constants[a]*constants[a]
    for a in range(p):
        for b in range(a+1,p):
            if S[a,b] > 0:
                Cab = sample_Cov(X[a,b][0],X[a,b][1],MM[a],MM[b])
                V[a,b] = constants[a]*constants[b]*Cab
                V[b,a] = constants[a]*constants[b]*Cab
            else:
                V[a,b] = 0
                V[b,a] = 0
    return graph(V)




def bayes_Prob(xx):
    # return Bayesian probability of xx
    x0 = xx.count(1)
    x1 = xx.count(-1)
    return (x0+1)/(x0+x1+2)

def bayes_Mean(xx):
    # return Bayesian mean of xx
    # x0 = xx.count(1)
    # x1 = xx.count(-1)
    # return (x0-x1)/(x0+x1+2)
    n = xx.count(1)-xx.count(-1)
    d = len(xx)+2
    return n/d

def bayes_Var(xx,Mx):
    # return Bayesian variance of xx with mean Mxx
    # x0 = xx.count(1)
    # x1 = xx.count(-1)
    # return 4*(x0+1)/(x0+x1+2)*(x1+1)/(x0+x1+2)
    n = ((xx.count(1)+1)*(xx.count(-1)+1))<<2
    d = (len(xx)+2)**2
    return n/d

def bayes_Cov(xx,yy,Px,Py):
    # return Bayesian covariance of xx,yy with means Mxx,Myy
    xxx = [(x,y) for x,y in zip(xx,yy)]
    # x00 = xxx.count((1,1))
    # x01 = xxx.count((1,-1))
    # x10 = xxx.count((-1,1))
    # x11 = xxx.count((-1,-1))
    # p00 = Px*Py
    # p01 = Px*(1-Py)
    # p10 = (1-Px)*Py
    # p11 = (1-Px)*(1-Py)
    # return (x00-x01-x10+x11)/(x00+x01+x10+x11) - (p00-p01-p10+p11)
    n = (xxx.count((1,1)) + xxx.count((-1,-1))) - (xxx.count((1,-1)) + xxx.count((-1,1)))
    d = len(xx)
    p = (Px*Py + (1-Px)*(1-Py)) - (Px*(1-Py) + (1-Px)*Py)
    return n/d-p
    # n = (((xxx.count((1,1))+1)*(xxx.count((-1,-1))+1)) - ((xxx.count((1,-1))+1)*(xxx.count((-1,1))+1)))<<2
    # d = (len(xx)+2)**2
    # return n/d

def bayes_variance_graph(P,X,constants):
    # return Bayesian variance graph
    p = P.paulis()
    PP = [bayes_Prob(X[a,a][0]) for a in range(p)]
    V = np.zeros((p,p),dtype=float)
    for a in range(p):
        if len(X[a,a][0]) > 0:
            V[a,a] = (constants[a]**2)*bayes_Var(X[a,a][0],bayes_Mean(X[a,a][0]))
        else:
            V[a,a] = (constants[a]**2)
    for a0,a1 in itertools.combinations(range(p),2):
        if len(X[a0,a1][0]) > 0:
            V01 = constants[a0]*constants[a1]*bayes_Cov(X[a0,a1][0],X[a0,a1][1],PP[a0],PP[a1])
            V[a0,a1] = V01
            V[a1,a0] = V01
        else:
            V[a0,a1] = 0
            V[a1,a0] = 0
    return graph(V)

def ready_to_measure_bvg(P,X,constants):
    # return Bayesian variance graph
    p = P.paulis()
    PP = [bayes_Prob(X[a,a][0]) for a in range(p)]
    S0 = np.array([[len(X[a,b][0]) for b in range(p)] for a in range(p)],dtype=int)
    S1 = np.array([[((S0[a,b])/(S0[a,a]*S0[b,b])-(S0[a,b]+1)/((S0[a,a]+1)*(S0[b,b]+1))) if S0[a,b]>0 else 0 for b in range(p)] for a in range(p)])
    V = np.zeros((p,p),dtype=float)
    for a in range(p):
        if S0[a,a] > 0:
            V[a,a] = (constants[a]**2)*bayes_Var(X[a,a][0],bayes_Mean(X[a,a][0]))*S1[a,a]
        else:
            V[a,a] = constants[a]**2
    for a0,a1 in itertools.combinations(range(p),2):
        if S0[a0,a1] > 0:
            V01 = constants[a0]*constants[a1]*bayes_Cov(X[a0,a1][0],X[a0,a1][1],PP[a0],PP[a1])*S1[a0,a1]
            V[a0,a1] = V01
            V[a1,a0] = V01
        else:
            V[a0,a1] = 0
            V[a1,a0] = 0
    return V





def progress_scaling(a):
    # Scales the progress bar appropriately
    return math.floor(a**(4/3))

def printProgressBar (iteration,total,length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration - Required : current iteration (Int)
        total     - Required : total iterations (Int)
        length    - Optional : character length of bar (Int)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '█' * filledLength + '-' * (length - filledLength)
    print(f' |{bar}| {percent}%', end="\r")
    # Print New Line on Complete
    if iteration == total: 
        print(" "*(length+10),end="\r")




# Read Hamiltonians

def read_Hamiltonian(path):
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

def write_maximal_cliques(path,aaa):
    f = open(path,"w")
    ll = ""
    for aa in aaa:
        l = ""
        for a in aa:
            l += (str(a)+" ")
        l = l[:-1]
        ll += (l+"\n")
    ll = ll[:-1]
    f.write(ll)
    f.close()

def read_maximal_cliques(path):
    f = open(path,"r")
    ll = f.readlines()
    f.close()
    aaa = []
    for l in ll:
        aaa.append([int(s) for s in l.split(" ")])
    return aaa

def write_ground_state(path,psi):
    ss = list(psi[a] for a in range(len(psi)))
    f = open(path,"w")
    for s in ss[:-1]:
        f.write(str(s)+" ")
    f.write(str(ss[-1]))
    f.close()

def read_ground_state(path):
    f = open(path,"r")
    ll = f.readlines()
    f.close()
    psi = []
    for l in ll:
        psi.append([np.complex(s) for s in l.split(" ")])
    return np.array(*psi)

def write_variance_graph(path,A):
    sss = list(list(A.adj[a1,a0] for a0 in range(A.ord())) for a1 in range(A.ord()))
    f = open(path,"w")
    for ss in sss[:-1]:
        for s in ss[:-1]:
            f.write(str(s)+" ")
        f.write(str(ss[-1])+"\n")
    for s in sss[-1][:-1]:
        f.write(str(s)+" ")
    f.write(str(sss[-1][-1]))
    f.close()

def read_variance_graph(path):
    f = open(path,"r")
    ll = f.readlines()
    f.close()
    adj_mat = []
    for l in ll:
        adj_mat.append([float(s) for s in l.split(" ")])
    return graph(np.array(adj_mat))
