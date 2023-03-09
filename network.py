import viscosity
import skimming
import numpy as np
import matplotlib.pyplot as plt
from ttd import *
from scipy.optimize import fsolve
import scipy.io
import pickle


class Network:
    def __init__(self,params):
        self.seed = None
        for key in params:
            setattr(self, key, params[key])
            
        self.params = params
        
        self.set_adj_inc() # compute adjacency and incidence matrices
        self.set_bh() # compute B(H) matrix
        
        # state vector is split into two pieces...
        self.pq = (self.nNodes + self.nVessels)*[np.nan] # ... [pressures; flows] and ...
        self.h = self.nVessels*[np.nan] # ... [hematocrits]
        self.equilibria = []

    def set_adj_inc(self):
        n_v = len(self.v)
        n_e = len(self.e)
        adj = np.zeros((n_v,n_v))
        inc = np.zeros((n_v,n_e))

        for i,edge in enumerate(self.e):
            e0 = min(edge)
            e1 = max(edge)

            adj[e0,e1] = 1
            adj[e1,e0] = 1

            # directed edge from lower index to higher index
            inc[e0,i] = -1
            inc[e1,i] = 1
            
        self.adj = adj
        self.inc = inc
        
        self.nNodes = int(adj.shape[0])
        self.nVessels = int(inc.shape[1])

        self.interiorNodes = np.where(np.sum(adj,axis=0) == 3)[0]
        self.exteriorNodes = np.where(np.sum(adj,axis=0) == 1)[0]
        self.exteriorFlows = np.where(np.sum(np.abs(inc[self.exteriorNodes,:]),axis=0) == 1)[0]         
        
    def set_bh(self,bc_kind='pressure'):
        poiseuille = [self.inc.T, np.zeros((self.nVessels,self.nVessels))]
        interior_flow_balance = [np.zeros((len(self.interiorNodes),self.nNodes)),self.inc[self.interiorNodes,:]]
        boundary_conditions = np.zeros((len(self.exteriorNodes),self.nVessels + self.nNodes))
        if bc_kind == 'pressure':
            for i,node in enumerate(self.exteriorNodes):
                boundary_conditions[i,node] = 1
            tail = self.pq_bcs[self.exteriorNodes]
        elif bc_kind == 'flow':
            boundary_conditions = N.inc[N.exteriorNodes,:]
            tail = self.pq_bcs[self.exteriorFlows]
            
        self.bh = np.block([poiseuille, interior_flow_balance, [boundary_conditions]])
        self.bh_rhs = np.concatenate((np.zeros(self.nNodes + self.nVessels - len(self.exteriorNodes)), tail))                 
        
    def set_cq(self,pq):
        q = pq[self.nNodes:]
        rhs = np.zeros(self.nVessels)

        X = self.inc@np.diag(np.sign(q))
        exterior = np.array((np.abs(X).sum(axis=1) == 1)).flatten()    
        interior = np.array((np.abs(X).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        net_in = np.array(X.sum(axis=1) == 1).flatten()    
        
        CQ = self.inc[self.interiorNodes,:]@np.diag(q)
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        for node in div:
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
            row = np.zeros(self.nVessels)
            row[idx_in] = -skimming.skimming_kj(np.abs(q[idx_out]/q[idx_in]),self.pPlasma)[0]
            row[idx_out] = 1
            CQ = np.concatenate((CQ,row.reshape(1,-1)))  
        
        inlets =  np.where(np.bitwise_and(exterior,net_out))[0]       
        for i,inlet in enumerate(inlets):
            row = np.zeros(self.nVessels)
            inlet_vessel = np.where(self.inc[inlet])[0][0]            
            
            row[inlet_vessel] = 1
            CQ = np.concatenate((CQ,row.reshape(1,-1)))              
            
            rhs[-(len(inlets)-i)] = self.h_bcs[inlet_vessel]

        self.cq = CQ
        self.cq_rhs = rhs 
        self._curr_q = q
        self._curr_p = pq[:self.nNodes]
    
    def get_pq(self,h):
        mat = self.bh.copy()
        r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        for i in range(self.nVessels):
            mat[i,self.nNodes+i] = r[i]        
        
        return np.linalg.solve(mat,self.bh_rhs)
    
    def set_state(self,h):
        self.h = h
        self.pq = self.get_pq(h)
    
    def equilibrium_relation(self,h):
        self._curr_h = h
        pq = self.get_pq(h)
        self.set_cq(pq)
        
        return h - np.linalg.solve(self.cq,self.cq_rhs)
                
    def find_equilibria(self,n,tol=1e-4,verbose=False):
        if self.seed: np.random.seed(self.seed)
        self.equilibria = []
        i = 0 
        while i < n:
            self.h = np.random.random(self.nVessels)
            
            try:
                outcome = fsolve(self.equilibrium_relation, self.h)
            except:
                continue
            residual = np.linalg.norm(self.equilibrium_relation(outcome))/len(outcome)
            if residual > tol:
                continue
        
            self.h = outcome
             
            if verbose:
                print(f'{i+1}: |F(x*)| = {np.round(residual,6)}')
            if i == 0:
                self.equilibria = self.h.reshape(-1,1)
            else:
                self.equilibria = np.concatenate((self.equilibria,self.h.reshape(-1,1)),axis=1)
            i += 1

    def directed_adj_dict(self):
        q = self.pq[self.nNodes:]
        A = {}
        for i in range(len(q)):
            v0 = min(self.e[i])
            v1 = max(self.e[i])
            if q[i] > 0:
                A[v0] = A.get(v0,[]) + [v1]
            else:
                A[v1] = A.get(v1,[]) + [v0]

        for key,item in A.items():
            A[key] = set(item)

        for i in set(range(int(self.adj.shape[0]))) - set(A.keys()):
            A[i] = set([])

        self._adj_dict = A
     
    def get_paths_by_node_from_inlet(self,inlet):
        stack = [(inlet,[inlet])]
        paths = []
        while stack:
            (vertex, path) = stack.pop()

            if len(self._adj_dict[vertex]) == 0:
                paths.append(path)
            for next in self._adj_dict[vertex] - set(path):
                stack.append((next, path + [next]))            
    
        return paths
       
    def get_paths_by_node(self):
        self.directed_adj_dict()
        paths = []
        for i in self.exteriorNodes:
            new = self.get_paths_by_node_from_inlet(i)
            if len(new[0]) == 1:
                continue
            paths += new   
        self._paths_by_node = paths
    
    def get_paths_by_edge(self):
        self.get_paths_by_node()
        self._paths_by_edge = [[np.abs(self.inc)[[path[i],path[i+1]],:].sum(axis=0).argmax() for i in range(len(path)-1)] for path in self._paths_by_node]        

    def compute_conditional_probabilities_downstream(self):
        q = self.pq[self.nNodes:]
        self._rbc = self.h*np.abs(q)   

        rbc_normalizer = np.zeros(self._rbc.shape)

        X = self.inc@np.diag(np.sign(q))
        for node in range(self.nNodes):
            row = X[node,:]
            if np.abs(row).sum() == 1:
                if row.sum() == -1:
                    vessel = np.where(row)[0][0]
                    rbc_normalizer[vessel] = np.sum(self._rbc[self.exteriorFlows])/2
            else:
                if row.sum() == -1:
                    inflow = np.where(row == 1)[0][0]
                    outflow_0 = np.where(row == -1)[0][0]
                    outflow_1 = np.where(row == -1)[0][1]                    
                    rbc_normalizer[outflow_0] = self._rbc[inflow]
                    rbc_normalizer[outflow_1] = self._rbc[inflow]                    
                elif row.sum() == 1:
                    outflow = np.where(row == -1)[0][0]
                    rbc_normalizer[outflow] = self._rbc[outflow]
        
        self._cond_prob = self._rbc/rbc_normalizer        
        
    def compute_ttd(self,verbose=False):
        p = self.pq[:self.nNodes]
        q = self.pq[self.nNodes:]

        self.compute_conditional_probabilities_downstream()
        if verbose:
            print(f'Cond prob: {self._cond_prob}')
        
        self.get_paths_by_edge()
        
        probs = []
        for path in self._paths_by_edge:
            probs.append(np.product(self._cond_prob[path]))

        if verbose: 
            for i,prob in enumerate(probs):
                print(f'P(RBC -> {self._paths_by_edge[i]}) \t= {np.round(prob,6)}')
            print(f'Check sum of total probability : {np.sum(probs)}')

        vol = np.pi*(self.d/2)**2*self.l
        tau = vol/np.abs(q)

        delays = []
        for path in self._paths_by_edge:
            delays.append(np.sum(tau[path]))

        if verbose: 
            for i,delay in enumerate(delays):
                print(f'{self._paths_by_edge[i]} :\t {delay}')
                
        ttd = TransitTimeDistribution(self._paths_by_edge, delays, probs)
#         if np.abs(np.sum(ttd.probs) - 1) > 1e-3:
#             print('Warning: Cumul. prob. of candidate TTD is not equal to 1!')
        self.ttd = ttd
    
    def plot(self,width=[],colors=[],directions=[],annotate=False,ms=10):
        x_min = np.min(self.v[:,0])-0.25
        x_max = np.max(self.v[:,0])+0.25
        y_min = np.min(self.v[:,1])-0.25
        y_max = np.max(self.v[:,1])+0.25
        if len(colors) == 0: colors = len(self.e)*['k']
        if len(width) == 0: width = len(self.e)*[1]
        for i,edge in enumerate(self.e):
            i0 = min([edge[0],edge[1]])
            i1 = max([edge[0],edge[1]])        

            x0 = self.v[i0,0]
            y0 = self.v[i0,1]

            x1 = self.v[i1,0]
            y1 = self.v[i1,1]

            if not self.w[i]:
                plt.plot([x0, x1], [y0,y1],'-', c=colors[i], lw=width[i])
            else:
                if x1 > x0:
                    plt.plot([x0, x1-(x_max-x_min)], [y0,y1], '-', c=colors[i], lw=width[i])
                    plt.plot([x0+(x_max-x_min), x1], [y0,y1], '-', c=colors[i], lw=width[i])
                else:
                    plt.plot([x0-(x_max-x_min), x1], [y0,y1], '-', c=colors[i], lw=width[i])
                    plt.plot([x0, x1+(x_max-x_min)], [y0,y1], '-', c=colors[i], lw=width[i])                
            if len(directions):
                if directions[i] == 1:
                    if len(self.w):
                        if self.w[i] == 0:
                            plt.arrow(x0,y0,(x1-x0)/2,(y1-y0)/2,head_width=.2,lw=0)                    
                elif directions[i] == -1:
                    if len(self.w):
                        if self.w[i] == 0:
                            plt.arrow(x1,y1,(x0-x1)/2,(y0-y1)/2,head_width=.2,lw=0)

            if annotate == True: 
                if self.w[i] == 0:
                    plt.annotate(str(i),(x0+(x1-x0)/2+0.05,y0+(y1-y0)/2),fontsize=16)
                else:
                    plt.annotate(str(i),(np.max((x0,x1))+.05,y1),fontsize=16,color='r')

        for node in range(len(self.v)):
            x = self.v[node,0]
            y = self.v[node,1]
            plt.plot(x,y ,'wo',mec='k',ms=ms)
            x,y = self.v[node,0], self.v[node, 1]
            if annotate:
                plt.text(x+0.02,y,str(node),fontsize=12)

        plt.gca().set_xticks([])
        plt.gca().set_yticks([])    

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)        

        plt.gca().set_aspect('equal')                

    def save(self,prefix):
        scipy.io.savemat(f'data/{prefix}_eqs.mat',{'eqs':self.equilibria})
        f = open(f'data/{prefix}_params.p','wb')
        pickle.dump(self.params,f)
        f.close()