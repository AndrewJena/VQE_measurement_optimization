import numpy as np
import itertools
import random
import time
import math


# NAMING CONVENTIONS

# a,b,c are integers/indices
# aa,bb,cc are lists of integers
# aaa,bbb,ccc are lists of lists of integers
# s,ss,sss are the same for strings
# m,mm,mmm are the same for matrices
# v,vv,vvv are the same for vertices
# P1,P2 are paulis
# G1,G2 are gates
# A1,A2 are graphs (i.e. adjacency matrices)
# p is number of paulis
# q is number of qubits
# d is number of interactions (this was Luca's convention)



# DEFINITIONS

# I,X,Y,Z Pauli matrices used for constructing tensor products 
I_mat = np.array([[1,0],[0,1]],dtype=complex)
X_mat = np.array([[0,1],[1,0]],dtype=complex)
Y_mat = np.array([[0,-1j],[1j,0]],dtype=complex)
Z_mat = np.array([[1,0],[0,-1]],dtype=complex)


def remove_subsets(aaa):
	tbr = set([])
	for aa1,aa2 in itertools.combinations(aaa,2):
		if aa1 < aa2:
			tbr.add(aa1)
		elif aa2 < aa1:
			tbr.add(aa2)
	return aaa-tbr


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

	def remove(self,aa):
		# Input: a Int
		# Output: Pauli without entry a
		for a in sorted(aa,reverse=True):
			self.X = np.delete(self.X,a,axis=0)
			self.Z = np.delete(self.Z,a,axis=0)

	def X_weight(self):
		# Input: none
		# Output: number of nonzero X entries
		if self.paulis() != 1:
			raise Exception("X weight can only be determined for a single Pauli")
		return np.count_nonzero(self.X)

def pauli_to_matrix(P):
	# Input: P pauli (a single Pauli rather than a list)
	# Output: matrix representing tensor product of Pauli matrices
	if P.paulis() != 1:
		raise Exception("Matrix can only be constructed for a single Pauli")
	X,Z = P.X[0],P.Z[0]
	mm = []
	for a in range(P.qubits()):
		if X[a]:
			if Z[a]:
				mm.append(Y_mat)
			else:
				mm.append(X_mat)
		else:
			if Z[a]:
				mm.append(Z_mat)
			else:
				mm.append(I_mat)
	return tensor(mm)

def print_pauli(P):
	# Input: P pauli
		# Print list of Paulis in symplectic form
	X,Z = P.X,P.Z
	for a in range(P.paulis()):
		print(P.X[a,:].astype(int),P.Z[a,:].astype(int))
	print()

def print_pauli_string(P):
	# Input: P pauli
		# Print list of Paulis in string form
	X,Z = P.X,P.Z
	for a in range(P.paulis()):
		for b in range(P.qubits()):
			if X[a,b]:
				if Z[a,b]:
					print("Y",end="")
				else:
					print("X",end="")
			else:
				if Z[a,b]:
					print("Z",end="")
				else:
					print("I",end="")
		print()

def string_to_pauli(sss):
	# Input: sss list of strings
	# Output: pauli object
	X = np.array([[s in set(["X","Y"]) for s in ss] for ss in sss],dtype=bool)
	Z = np.array([[s in set(["Z","Y"]) for s in ss] for ss in sss],dtype=bool)
	return pauli(X,Z)

def symplectic_ip(P1,P2):
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



# GATES

class gate:
	# Input: name function; qubits list of targets
	def __init__(self,name,qubits):
		self.name = name
		self.qubits = qubits

def act(P,G):
	# Input: P pauli; G gate
	# Output: result of G acting on P (by conjugation)
	return G.name(P,G.qubits)

def H(P,aa):
	# Input: P pauli; aa list of Ints (target,)
	# Output: result of Hadamard acting on P on qubit aa[0]
		# X -> Z
		# Z -> X
	X,Z = P.X,P.Z
	a0 = aa[0]
	X[:,a0],Z[:,a0] = Z[:,a0].copy(),X[:,a0].copy()
	return pauli(X,Z)

def S(P,aa):
	# Input: P pauli; aa list of Ints (target,)
	# Output: result of S gate (phase gate) acting on P on qubit aa[0]
		# X -> Y
		# Z -> Z
	X,Z = P.X,P.Z
	a0 = aa[0]
	Z[:,a0] ^= X[:,a0]
	return pauli(X,Z)

def CNOT(P,aa):
	# Input: P pauli; aa list of Ints (control,target)
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
	# Input: P pauli; aa list of Ints (target,target)
	# Output: result of CZ acting on P on qubits aa[0],aa[1]
		# XI -> XZ
		# IX -> ZX
		# ZI -> ZI
		# IZ -> IZ
	X,Z = P.X,P.Z
	a0,a1 = aa[0],aa[1]
	Z[:,a1] ^= X[:,a2]
	Z[:,a2] ^= X[:,a1]
	return pauli(X,Z)

def SWAP(P,aa):
	# Input: P pauli; aa list of Ints (target,target)
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

def iSWAP(P,aa):
	# Input: P pauli; aa list of Ints (target,target)
	# Output: result of iSWAP acting on P on qubits aa[0],aa[1]
		# XI -> ZY
		# IX -> YZ
		# ZI -> IZ
		# IZ -> ZI
	X,Z = P.X,P.Z
	a0,a1 = aa[0],aa[1]
	X[:,a1],X[:,a2] = X[:,a2].copy(),X[:,a1].copy()
	Z[:,a1],Z[:,a2] = Z[:,a2].copy(),Z[:,a1].copy()
	Z[:,a1] ^= (X[:,a1]^X[:,a2])
	Z[:,a2] ^= (X[:,a1]^X[:,a2])
	return pauli(X,Z)



# DIAGONALIZATION



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

def print_graph(A):
	# Input: A graph
		# Print adjacency matrix of graph
	print(A.adj)
	print()

def print_graph_neighbors(A):
	# Input: A graph
		# Print vertex: neighbors view of graph
	for v in range(A.ord()):
		print(v,end=": ")
		for w in A.neighbors(v):
			print(w,end=" ")
		print()

def max_covariance_allowed(A,c):
	B = graph(np.copy(A.adj))
	for v1,v2 in itertools.combinations(range(B.ord()),2):
		if B.adj[v1,v2] > c:
			B.lade_edge(v1,v2,0)
	return B



# GRAPH ALGORITHMS

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

def greedy_min_shots(A,error=None,shots=None,shot_step=1):
	# Input: A graph
	# Output: list of lists of Ints (the partition into cliques)
		# Greedy algorithm with random vertex ordering
		# Sorting vertices may improve algorithm, but only slightly
	p = A.ord()
	fewest_shots = cost(A,[[a] for a in range(p)],error=error,shots=shots,shot_step=shot_step)
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
				ddd = aaa.copy()
				ddd[d] = dd
				s = cost(A,ddd+[[bb[e]] for e in range(c+1,p)],error=error,shots=shots,shot_step=shot_step)
				if s < fewest_shots:
					fewest_shots = s
					min_part = aa
					flag = True
		if flag:
			aaa.remove(min_part)
			aaa.append(min_part+[bb[c]])
		else:
			aaa.append([bb[c]])
	return aaa

def anneal_min_shots(A,error=None,shots=None,shot_step=1):
	# Input: A graph
	# Output: list of lists of Ints (the partition into cliques)
		# Deterministic annealing-type algorithm, where cliques are merged at each step
	aaa = [[a] for a in range(A.ord())]
	fewest_shots = cost(A,aaa,error=error,shots=shots,shot_step=shot_step)
	flag = True
	while flag:
		flag = False
		aaa_new = aaa.copy()
		for aa1,aa2 in itertools.combinations(aaa,2):
			bb = aa1+aa2
			if A.clique(bb):
				bbb = aaa.copy()
				bbb.remove(aa1)
				bbb.remove(aa2)
				bbb.append(bb)
				s = cost(A,bbb,error=error,shots=shots,shot_step=shot_step)
				if s < fewest_shots:
					fewest_shots = s
					aaa_new = bbb
					flag = True
		if flag:
			aaa = aaa_new
	return aaa

def exhaust_min_shots(A,error=None,shots=None,shot_step=1):
	# Input: A graph
	# Output: list of lists of Ints (the partition into cliques)
		# Just initializes (and returns output of) recursive algorithm
		# Exhaustively searches through all possible partitions to find optimal
	aaa = [[a] for a in range(A.ord())]
	fewest_shots = cost(A,aaa,error=error,shots=shots,shot_step=shot_step)
	best_partition = aaa
	return exhaust_min_shots_rec(A,aaa,fewest_shots,best_partition,error=error,shots=shots,shot_step=shot_step)[1]

def exhaust_min_shots_rec(A,aaa,fewest_shots,best_partition,error=None,shots=None,shot_step=1):
	# Input: A graph; aaa list of lists of Ints; fewest_shots real; best_partition list of lists of Ints
	# Output: fewest_shots and best_partition
		# Recursively iterates through all possible partitions and outputs optimal partition and number of shots
	s = cost(A,aaa,error=error,shots=shots,shot_step=shot_step)
	if fewest_shots-s > 2**(-63):
		fewest_shots = s
		best_partition = aaa
	for aa1,aa2 in itertools.combinations(aaa,2):
		bb = aa1+aa2
		if A.clique(bb):
			bbb = aaa.copy()
			bbb.remove(aa1)
			bbb.remove(aa2)
			bbb.append(bb)
			s,ccc = exhaust_min_shots_rec(A,bbb,fewest_shots,best_partition,error=error,shots=shots,shot_step=shot_step)
			if s < fewest_shots:
				fewest_shots = s
				best_partition = ccc
	return fewest_shots,best_partition

def greedy_edge_clique_cover(A):
	# ANNEAL IS WORSE THAN GREEDY (BOTH TIME AND NUMBER OF CLIQUES)
	p = A.ord()
	aaa = [[v] for v in range(p) if A.degree(v) == 1]
	remaining_edges = set([frozenset(vs) for vs in itertools.combinations(range(p),2) if A.clique(vs)])
	while remaining_edges:
		aa = remaining_edges.pop()
		remaining_vertices = set(range(p))-aa
		for v in remaining_vertices:
			bb = aa|set([v])
			if A.clique(bb) and any(frozenset([a,v]) in remaining_edges for a in aa):
				aa = bb
		aaa.append(list(aa))
		for v1,v2 in itertools.combinations(aa,2):
			remaining_edges.discard(frozenset([v1,v2]))
	return aaa

def maximal_cliques(A):
	# return all maximal cliques of A
	p = A.ord()
	aaa = set([frozenset([0])])
	for a in range(1,p):
		aset = set([a])
		inter = A.neighbors(a)
		aaa |= set([frozenset(aset|(inter&aa)) for aa in aaa])
		aaa = remove_subsets(aaa)
	return list([list(aa) for aa in aaa])

def clique_sort_func(A,aa):
	C = A.adj
	return sum(sum(C[a,b] for b in aa) for a in aa) - sum(C[a,a] for a in aa)

def maximal_cliques_sorted_cover(A):
	p = A.ord()
	aaa = maximal_cliques(A)
	aaa.sort(key=lambda aa: clique_sort_func(A,aa))
	bbb = []
	while set([b for b in range(p) if not (b in itertools.chain(*bbb))]):
		aa = aaa.pop(0)
		if clique_sort_func(A,aa) < 0:
			bbb.append(aa)
		else:
			break
	return bbb



# PHYSICS FUNCTIONS

def Var(P,psi):
	# Input: P pauli; psi np.array
	# Output: \langle psi | P^2 | psi \rangle - (\langle psi | P | psi \rangle)^2
	m = pauli_to_matrix(P)
	psi_dag = psi.conj().T
	var = (psi_dag.dot(m).dot(m).dot(psi))-(psi_dag.dot(m).dot(psi))**2
	return var

def Cov(P1,P2,psi):
	# Input: P1,P2 paulis; psi np.array
	# Output: \langle psi | P Q | psi \rangle - (\langle psi | P | psi \rangle)(\langle psi | Q | psi \rangle)
	m1 = pauli_to_matrix(P1)
	m2 = pauli_to_matrix(P2)
	psi_dag = psi.conj().T
	cov = (psi_dag.dot(m1).dot(m2).dot(psi))-(psi_dag.dot(m1).dot(psi))*(psi_dag.dot(m2).dot(psi))
	return cov

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
			if not symplectic_ip(Pa,Pb):
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
			if not symplectic_ip(Pa,Pb):
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
			if symplectic_ip(Pa,Pb):
				A.lade_edge(a,b,1)
	return A

def very_random_Ham(p,q,d):
	# Input: p,q,d Ints (# Paulis, # qubits, max weight)
	# Output: random pauli satisfying input conditions
		# This Hamiltonian generation algorithm was stolen from Luca's code
	string_list = []
	for i in range(p):
		s = ["I" for _ in range(q)]
		for j in random.sample(range(q),d):
			c = random.randint(0,3)
			if c == 1:
				s[j] = "Z"
			elif c == 2:
				s[j] = "X"
			elif c == 3:
				s[j] = "Y"
		string_list.append("".join(s))
	return string_to_pauli(string_list)

def ground_state(m):
	# Input: m np.array
	# Output: eigenvector corresponding to lowest eigenvalue (in absolute value)
	vals,vecs = np.linalg.eigh(m)
	return np.array([v for v in vecs[:,np.argmin([abs(e) for e in vals])]],dtype=complex)

def tensor(mm):
	# Input: list of matrices
	# Output: tensor product of matrices
	if len(mm) == 0 :
		return matrix([])
	elif len(mm) == 1 :
		return mm[0]
	else:
		return np.kron(mm[0],tensor(mm[1:]))

def print_Ham_string(P,constants):
	# Input: P pauli
		# Print list of Paulis in string form together with constants
	X,Z = P.X,P.Z
	for a in range(P.paulis()):
		for b in range(P.qubits()):
			if X[a,b]:
				if Z[a,b]:
					print("Y",end="")
				else:
					print("X",end="")
			else:
				if Z[a,b]:
					print("Z",end="")
				else:
					print("I",end="")
		if constants[a] >= 0:
			print(" +%s"%constants[a])
		else:
			print(" %s"%constants[a])

def smart_divide(a,b):
	if b == 0:
		return a
	else:
		return a/b

def Hadamard_divide(m,n):
	# return Hadamard quotient of m and n
	return np.vectorize(smart_divide)(m,n)

def Hadamard_multiply(m,n):
	# return Hadamard product of m and n
	return np.multiply(m,n)

def cost_Var(CZMZ,EMj):
	# return epsilon^2
	return np.sum(Hadamard_divide(CZMZ,EMj.dot(EMj.T)))

def cost(A,aaa,error=None,shots=None,Z=None,shot_step=1):
	# bucket filling algorithm for calculating cost
	# return cost as (number of measurements)*(sample variance)
	p = A.ord()
	bbb = aaa+[[b] for b in range(p) if not (b in itertools.chain(*aaa))]
	r = len(bbb)
	C = A.adj
	M = shot_step*np.identity(r)
	j = np.ones((r,1))
	E = np.zeros((p,r))
	for c1 in range(p):
		for c2 in range(r):
			if c1 in bbb[c2]:
				E[c1,c2] = 1
	Et = E.T
	if Z == None:
		Z,Zt = E,Et
	else:
		Zt = Z.T
	CZMZ = Hadamard_multiply(C,Z.dot(M).dot(Zt))
	CZMZ_plus = []
	for c1 in range(r):
		I = np.zeros((r,r))
		I[c1,c1] = shot_step
		CZMZ_plus.append(Hadamard_multiply(C,Z.dot(I).dot(Zt)))
	EMj = E.dot(M).dot(j)
	EMj_plus = []
	for c1 in range(r):
		I = np.zeros((r,r))
		I[c1,c1] = shot_step
		EMj_plus.append(E.dot(I).dot(j))
	V = cost_Var(CZMZ,EMj)
	s = shot_step*r
	if (shots != None):
		for s in range(shot_step*(r+1),shots+1,shot_step):
			VV = [cost_Var(CZMZ+CZMZ_plus[c],EMj+EMj_plus[c]) for c in range(r)]
			bucket = np.argmin(VV)
			V = VV[bucket]
			CZMZ = CZMZ+CZMZ_plus[bucket]
			EMj = EMj+EMj_plus[bucket]
		return s*V
	if (error != None):
		while V > error:
			VV = [cost_Var(CZMZ+CZMZ_plus[c],EMj+EMj_plus[c]) for c in range(r)]
			bucket = np.argmin(VV)
			V = VV[bucket]
			CZMZ = CZMZ+CZMZ_plus[bucket]
			EMj = EMj+EMj_plus[bucket]
			s += shot_step
		return s*V
	else:
		raise Exception("Must define either error or shots")

def cost_allow_zero(A,aaa,error=None,shots=None,Z=None,shot_step=1):
	# bucket filling algorithm for calculating cost
		# begin by giving 1/1000000 measurements to each Pauli
	# return cost as (number of measurements)*(sample variance)
	p = A.ord()
	bbb = [[b] for b in range(p)]+aaa
	r = len(bbb)
	C = A.adj
	M = np.zeros((r,r))
	for c in range(p):
		M[c,c] = 1/1000000
	j = np.ones((r,1))
	E = np.zeros((p,r))
	for c1 in range(p):
		for c2 in range(r):
			if c1 in bbb[c2]:
				E[c1,c2] = 1
	Et = E.T
	if Z == None:
		Z,Zt = E,Et
	else:
		Zt = Z.T
	CZMZ = Hadamard_multiply(C,Z.dot(M).dot(Zt))
	CZMZ_plus = []
	for c1 in range(r):
		I = np.zeros((r,r))
		I[c1,c1] = shot_step
		CZMZ_plus.append(Hadamard_multiply(C,Z.dot(I).dot(Zt)))
	EMj = E.dot(M).dot(j)
	EMj_plus = []
	for c1 in range(r):
		I = np.zeros((r,r))
		I[c1,c1] = shot_step
		EMj_plus.append(E.dot(I).dot(j))
	V = cost_Var(CZMZ,EMj)
	s = shot_step*p
	buckets = [0]*(r-p)
	if (shots != None):
		for s in range(shot_step*(p+1),shots+1,shot_step):
			VV = [cost_Var(CZMZ+CZMZ_plus[c+p],EMj+EMj_plus[c+p]) for c in range(r-p)]
			bucket = np.argmin(VV)
			buckets[bucket] += 1
			V = VV[bucket]
			bucket += p
			CZMZ = CZMZ+CZMZ_plus[bucket]
			EMj = EMj+EMj_plus[bucket]
		# print(buckets)
		# return s*V
		return buckets
	if (error != None):
		while V > error:
			VV = [cost_Var(CZMZ+CZMZ_plus[c],EMj+EMj_plus[c]) for c in range(r)]
			bucket = np.argmin(VV)
			V = VV[bucket]
			CZMZ = CZMZ+CZMZ_plus[bucket]
			EMj = EMj+EMj_plus[bucket]
			s += shot_step
		return s*V
	else:
		raise Exception("Must define either error or shots")


# WORKING CODE

# Runtime notes:
	# Pauli Matrix time grows linearly with p, exponentially with q (q=12 max)
	# Ground State time grows exponentially with q (q=12 max)
	# Variance Graph time grows quadratically with p, exponentially with q (p=1000,q=12 max)
	# Commutation Graph time grows quadratically with p (p=5000 max)
	# Greedy Minimum Parts time grows linearly with p (p=20000 max)
	# Greedy Minimum Shots grows linearly with p (p=20000 max)
	# Anneal Minimum Shots grows quadratically with p (p=1000 max)
	# Exhaust Minimum Shots grows like Bell number of p (p=12 max)


error = 0.01
shots = None
shot_step = 1
p = 8
q = 8
d = 8

P = very_random_Ham(p,q,d)
constants = [random.uniform(-1,1) for a in range(p)]
# print_Ham_string(P,constants)
print("Paulis:",p)
print("Qubits:",q)
print()

start = time.time()
m = sum(pauli_to_matrix(P.a_pauli(a))*constants[a] for a in range(p))
# print(m)
print("Pauli Matrix time:",time.time()-start)

start = time.time()
psi = ground_state(m)
# print(psi)
print("Ground State time:",time.time()-start)

start = time.time()
A = variance_graph(P,constants,psi)
# print_graph(A)
print("Variance Graph time:",time.time()-start)
print()


start = time.time()
aaa = greedy_min_parts(A)
# print(aaa)
print("Greedy Minimum Parts Algorithm")
print("Number of Parts:",len(aaa))
print("Runtime:",time.time()-start)
print("Cost:",cost(A,aaa,error=error,shots=shots))
print()

start = time.time()
aaa = greedy_min_shots(A,error=error,shots=shots)
# print(aaa)
print("Greedy Minimum Shots Algorithm")
print("Number of Parts:",len(aaa))
print("Runtime:",time.time()-start)
print("Cost:",cost(A,aaa,error=error,shots=shots))
print()

start = time.time()
aaa = anneal_min_shots(A,error=error,shots=shots)
# print(aaa)
print("Anneal Minimum Shots Algorithm")
print("Number of Parts:",len(aaa))
print("Runtime:",time.time()-start)
print("Cost:",cost(A,aaa,error=error,shots=shots))
print()

start = time.time()
aaa = greedy_edge_clique_cover(max_covariance_allowed(A,1))
# print(aaa)
print("Greedy Edge Clique Cover Algorithm")
print("Number of Parts:",len(aaa))
print("Runtime:",time.time()-start)
print("Cost:",cost(A,aaa,error=error,shots=shots))
print()

start = time.time()
aaa = maximal_cliques(A)
# print(aaa)
print("Maximal Cliques Algorithm")
print("Number of Parts:",len(aaa))
print("Runtime:",time.time()-start)
print("Cost:",cost(A,aaa,error=error,shots=shots))
print()

start = time.time()
aaa = maximal_cliques_sorted_cover(A)
# print(aaa)
print("Maximal Cliques Sorted Cover Algorithm")
print("Number of Parts:",len(aaa))
print("Runtime:",time.time()-start)
print("Cost:",cost(A,aaa,error=error,shots=shots))
print()



