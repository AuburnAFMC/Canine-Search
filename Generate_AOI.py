# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:42:18 2023
@author: Joseph Kennedy
"""

import numpy as np
import pandas as pd
import pydtmc as mc
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


class Bottle_Neck():
    def __init__(self, size, arrival, horizon ,decay = 0.9, material_con = 1,  **kwargs):
        self.r = decay
        self.d = material_con
        self.T = horizon
        #self.decay_mat = (1 - self.r)* np.eye(size)
        self.arrival = arrival

        
        if arrival == 'geometric':
            self.alpha = kwargs['ALPHA']
            self.trans_mat = self.generate_geometric_trans_mat(size)
            self.compute_nbrhd()
        if arrival == 'uniform':
            self.S = kwargs['S']
            self.trans_mat = self.generate_uniform_trans_mat(size) 
            self.compute_nbrhd()
            

        return
        
    
    
    def generate_geometric_trans_mat(self,size):
        '''
        Returns
        -------
        numpy array(sizeXsize)
            time homogeneous transition matrix for a bottle neck system with 
            geometric arrivals
        '''
        self.size = size+1
        
        #first row
        first = [1-self.alpha, self.alpha]
        for i in range(self.size - 2):
            first.append(0)
        first = np.array(first)
        
        #last row
        last = [0 for i in range(self.size-1)]
        last.insert(len(last),1)
        last = np.array(last)
        
        #middle right
        mid_right = np.eye(self.size-2)
        
        #middle left
        mid_left = np.zeros(shape = (self.size-2,2))
        # Restrict the auxillary cells
        self.Restricted = [self.size-1,0]
        return np.block([[first],[mid_left,mid_right],[last]])
    
    
    def generate_uniform_trans_mat(self,size):
        '''
     
        Returns
        -------
        numpy array(sizeXsize)
            time homogeneous transition matrix for a bottle neck system with 
            unifrom arrivals
        '''
        self.size = size
        # first row
        first = [0]
        for i in range(self.S):
            first.append(1/self.S)
        for i in range(self.size - 1):
            first.append(0)
        first = np.array(first)
        # last row
        last = [0 for i in range(self.size+self.S-1)]
        last.insert(len(last),1)
        last = np.array(last)
        # middle right
        mid_right = np.eye(self.size+self.S-2)
        # middle left
        mid_left = np.zeros(shape = (self.size+self.S-2,2))
        
        # Restrict the auxillary cells
        #self.Restricted = [i for i in range(0,self.S+1)]
        #self.Restricted.extend([self.size+self.S ]) #These are the queue states or the exit state
        self.Restricted = [self.size-1,0]
        #print(self.Restricted)
        return np.block([[first],[mid_left,mid_right],[last]])
    
    def make_chain(self):
        if self.arrival == 'uniform':
            chain_space_dim = self.size + self.S
            chain = mc.MarkovChain(self.trans_mat,[str(i) for i in range(chain_space_dim)]) # Markov Chain object
    
        if self.arrival == 'geometric':
            chain = mc.MarkovChain(self.trans_mat,[str(i) for i in range(self.size)])
        return chain
    def sample(self, samp_num, glimpse_func = 'exp', **kwargs):
        '''
        Samples random walks and computes detectable material deposit and 
        glimpse for the sample walks

        Parameters
        ----------
        samp_num : int
            number of smapled walks
        glimpse_func : string, optional
            Choice of glimpse function.
                exp - exponential
                tanh - hyperbolic tangent
            The default is 'exp'.

        Returns
        -------
        None.

        '''
        self.sample = [] # Sample list of walks
        self.walk = []
        self.samp_num = samp_num
        #Sample the state evolution 
        for j in tqdm(range(samp_num)):
            #initialize the state and sample paramters
            chain = self.make_chain()
            walk = ['0'] # Initial state
            if self.arrival == 'uniform':
                X_state = [self.d if i == 0 else 0 for i in range(self.size+self.S)]
            X_state = [self.d if i == 0 else 0 for i in range(self.size)] # Initial state 
            sample_walk = [X_state] # Sample of a single walk
            
            for i in range(1, self.T+1):
                #Get next state
                current_state = walk[-1]
                next_state = chain.next(current_state)#, seed = i*j)
                #if self.arrival  == 'uniform':
                #    if 
                #print(next)
                #Deposit and decay the material
                X_state = list(np.dot(np.array(X_state).T,self.r)+ self.d*np.array([1 if i == int(next_state) else 0 for i in range(self.size)]))
                #print(X_state)
                
                walk.append(next_state)
                sample_walk.append(X_state)
            if self.arrival == 'uniform':
                for i in range(len(walk)):
                    if int(walk[i]) <= self.S:
                        walk[i] = '0'
                    else:
                        walk[i] = str(int(walk[i])-self.size)
            self.sample.append(sample_walk)
            self.walk.append(walk)
        #print(self.walk)
        # generate the glimpse values for every sample    
        self.set_glimpse(glimpse_func, params = {i:kwargs[i] for i in kwargs})
        self.create_time_staged_graph()
        self.calculate_modified_graph() # must happen after create_time_staged_graph
        return
    
    
    def set_glimpse(self, glimpse_func, params):
        '''
        Returns the probability of detect given there is x amount of detectable material in the cell
        '''
        if glimpse_func == 'exp':
            try:
                self.glimpse_array =  1 - np.exp(-1*params.get('sweep',' ')*np.array(self.sample))
            except:
                print("parameters for the 'exp' glimpse has not been passed so glimpse values were not calculated")
        elif glimpse_func == 'tanh':
            self.glimpse_array =  np.tanh(np.array(self.sample))
        return
    
    def compute_nbrhd(self):
        self.nbrhd = {}
        visited = []
        for cell in range(len(self.trans_mat)):
            if cell not in visited and cell not in self.Restricted:
                # The first set is everything z points to, the second is everything that points to z the last set removes z from the set 
                if self.arrival == 'uniform':
                #    self.nbrhd[cell] =[i  for i in list(set(np.array([i for i in range(self.S, self.size)])[self.trans_mat[cell][self.S:self.size] > 0]).union([i  for i in range(self.S, self.size) if self.trans_mat[i][cell] > 0 or i == cell]) - set(self.Restricted))]
                    #self.nbrhd[cell] =list(set(np.array([i for i in range(self.size+self.S)])[self.trans_mat[cell] > 0]).union([i  for i in range(self.size+self.S) if self.trans_mat[i][cell] > 0 or i == cell]))# - set(self.Restricted))
                    if cell >= self.S:
                        
                        self.nbrhd[cell-self.S] = list(np.array([i for i in range(self.size)])[self.trans_mat[cell][self.S:]>0])
                        #print(cell-self.S,self.nbrhd[cell-self.S])
                        self.nbrhd[cell-self.S] = list(set(self.nbrhd[cell-self.S]).union(set([i  for i in range(self.size) if self.trans_mat[i+self.S][cell] > 0 or i+self.S == cell])))
                        #print(self.nbrhd[cell-self.S])
                if self.arrival == 'geometric':
                    self.nbrhd[cell] =list(set(np.array([i for i in range(self.size)])[self.trans_mat[cell] > 0]).union([i  for i in range(self.size) if self.trans_mat[i][cell] > 0 or i == cell]) - set(self.Restricted))
                
                #self.nbrhd[cell].extend([cell])
            visited.extend([cell])
        print(self.nbrhd)
        return    
    
    
    def expected_num_detections(self, state, time):
        '''
        Parameters
        ----------
        state : int
            The numeric representation of the cell being searched.
        time : int
            The time stage the cell is being searched in.

        Returns
        -------
        search independent expected number of detections in a single cell at a single time
        '''
        #average over the sample paths
        prob = 0
        for n in range(self.samp_num):
            #calculate the probability of success
            try:
                prob += self.glimpse_array[n][time][state]
            except:
                pass
        return (1/self.samp_num)*prob
    
    def expected_num_detections_modified(self, state, time):
        '''
        Parameters
        ----------
        state : int
            The numeric representation of the cell being searched.
        time : int
            The time stage the cell is being searched in.

        Returns
        -------
        search independent expected number of detections in a single cell at a single time
        '''
        
        
        #average over the sample paths
        prob = 0
        for n in range(self.samp_num):
            #calculate the probability of failure
            try:
                prob += self.glimpse_array[n][time][state]
            except:
                pass
        return (1/self.samp_num)*prob
    
    
 

    def Total_Prob_Det (self, z):
        pd_terms = [self.Prob_Det_t(z[:t+1]) for t in range(len(z))]
        val = sum(pd_terms)
        return [val,pd_terms]
    
    def Prob_Det_t(self,z):
        t = len(z[1:])
        pd = 0
        if len(z) > 1:
            for n in range(self.samp_num):
                # calculate the probability of failure

                prod = self.glimpse_array[n][t][z[t]]
                for u in range(1,t+1):
                    prod *= 1-self.glimpse_array[n][u][z[u]]
                #Calculate the posterior probability of detect in z[t] at time t given sample path n
                try:
                    pd += (1/self.samp_num)*prod
                except:
                    pass
        return pd   
     
    
    def create_time_staged_graph(self, G = None):
        '''
        Creates a staged-time graph for the bottleneck AOI and populates all 
        the parameters for weights and node positions
        '''
        # # Initialize the graph we will use to create the stage graph
        # adjacency = np.zeros((len(self.nbrhd ),len(self.nbrhd )))
        # for i in self.nbrhd:
        #     for j in self.nbrhd[i]:
        #         adjacency[i-1][j-1] = 1
        # print(adjacency)
        if G == None:
            source = []
            target = []
            added = []
            for cell in self.nbrhd:
                for nbr_cell in self.nbrhd[cell]:
                    if (cell,nbr_cell) not in added:
                        source.append(cell)
                        target.append(nbr_cell)
                        added.append((cell,nbr_cell))
            adj_lst = pd.DataFrame(data = {'source': source,'target': target})
            
            # create the network using networkx
            G = nx.from_pandas_edgelist(df = adj_lst, source = 'source', target = 'target')
        
        #print(list(G.edges()))
        # create the network using networkx
        #G = nx.from_numpy_matrix(adjacency)
        
        #create the staged graph
        self.Staged_Graph = nx.DiGraph()
        self.node_pos = {}
        # Add the first stage of nodes
        #print(self.Restricted)
        for i in G.nodes():
            if i not in self.Restricted:
                self.Staged_Graph.add_node((i,0))
                self.node_pos[(i,0)] = (0,i)
            
        # Add the rest of the stages and connect using the edge list
        for t in range(1,self.T + 1):
            for i in G.nodes():
                if i not in self.Restricted:
                    self.Staged_Graph.add_node((i,t))
                    self.node_pos[(i,t)] = (t,i)
            for edge in G.edges():
                if (edge[0] not in self.Restricted) and (edge[1] not in self.Restricted):
                    self.Staged_Graph.add_edge((edge[0],t-1),(edge[1],t), weight = -1*self.expected_num_detections(state = edge[1], time = t))
                    self.Staged_Graph.add_edge((edge[1],t-1),(edge[0],t), weight = -1*self.expected_num_detections(state = edge[0], time = t)) 
        self.Staged_Graph.add_node(('END',self.T+1))
        self.node_pos[('END',self.T + 1)] = (self.T + 1, len(list(G.nodes()))/2)
        for i in G.nodes():
            if i not in self.Restricted:
                self.Staged_Graph.add_edge((i,self.T),("END",self.T + 1),weight = 0)
            
        #Add the start node and attach
        self.Staged_Graph.add_node('start')
        self.node_pos['start'] = (-1,len(list(G.nodes()))/2)
        
        #Add the egdes to the first stage
        for i in G.nodes():
            if i not in self.Restricted:
                self.Staged_Graph.add_edge('start',(i,0),weight = 0)
        #print(list(self.Staged_Graph.edges()))
        return      
    
    
    def calculate_modified_graph(self, G = None):
        '''
        Create the time stage graph for the adjustment where the min of the 
        reverse start is applied to the node
        '''
        H = self.Staged_Graph.copy()
        weights = nx.get_edge_attributes(self.Staged_Graph, 'weight')#used to reset the weight of H
                #average over the sample paths
        
        for node in H.nodes():
            prob = 0
            for n in range(self.samp_num):
                #calculate the probability of success given the worst failure for this particular target path
                try:
                    #get the worst previous failure
                    min_ = 1
                    for nghbr in H.predecessors(node):
                        if self.glimpse_array[n][nghbr[1]][nghbr[0]] < min_:
                            min_ = self.glimpse_array[n][nghbr[1]][nghbr[0]]

                    prob += self.glimpse_array[n][node[1]][node[0]]* ( 1- min_)
                except:
                    pass
            for nghbr in H.predecessors(node):
                weights[(nghbr,node)] = -1*(1/self.samp_num)*prob
        # for node in H.nodes():
        #     weights = {}
        #     for nghbr in H.predecessors(node):
        #         #Get the expected prob detects of the neighbors and take the min
        #         #temp[nghbr] = self.expected_num_detections(state = nghbr[1], time = nghbr[0])
        #         weight[nghbr,node] = self.expected_num_detections_modified(state_cur = node[1],
        #                                                             time_cur = node[0],
        #                                                             state_prev = nghbr[1],
        #                                                             time_prev = nghbr[0])
        #     #if temp: #if temp is non empty
        #     #    min_ =min(temp, key=temp.get) #returns the Key corresponsiding to the min value
        #     #    #update the weights
        #     #    weights[(min_,node)] = weights[(min_,node)]*(1 - temp[min_])
        
         # needs to be reformatted to be set
        for edge in weights:
            weights[edge] = {'weight':weights[edge]} 
            
        nx.set_edge_attributes(H, weights)
        self.Staged_Graph_Modified = H.copy()
        return
    
    def plot_staged_graph(self, figsz =(30, 35),emphasize = False ):
        if emphasize:
            fig = plt.figure(1, figsize=figsz)
            nx.draw(self.Staged_Graph,pos = self.node_pos,node_size = 100, width = [max(-200*i,1) for i in nx.get_edge_attributes(self.Staged_Graph, 'weight').values()])
        else:
            fig = plt.figure(1, figsize=figsz)
            nx.draw(self.Staged_Graph,pos = self.node_pos,node_size = 100)
        
        return
    
    def plot_staged_graph_modified(self, figsz =(30, 35),emphasize = False ):
        if emphasize:
            fig = plt.figure(1, figsize=figsz)
            nx.draw(self.Staged_Graph_Modified,pos = self.node_pos,node_size = 100, width = [-100*i for i in nx.get_edge_attributes(self.Staged_Graph, 'weight').values()])
        else:
            fig = plt.figure(1, figsize=figsz)
            nx.draw(self.Staged_Graph_Modified,pos = self.node_pos,node_size = 100)
        
        return
    
    def count_search_paths(self):
        '''
        Counts the number of search paths 
        '''
        node_sub = []
        for i in list(self.Staged_Graph.nodes):
            try:
                if i[1] == 1 or i[1] == 2:
                    node_sub.append(i)
            except:
                pass
        
        Sub = self.Staged_Graph.subgraph(node_sub)
        A = np.zeros((int(len(node_sub)/2),int(len(node_sub)/2))) 
        for i in Sub.edges():
            A[i[0][0]-1][i[1][0]-1] = 1
            
        power = np.linalg.matrix_power(A,self.T)
        sum_ = 0
        for i in range(len(A)):
            for j in range(len(A)):
                sum_ += power[i][j] 
                
        return sum_

class Two_D_Hex_grid(Bottle_Neck):
    def __init__(self, size_x,size_y, lambda_ = 0.5, arrival = 'geometric', horizon = 10 ,decay = 0.9, material_con = 1,  **kwargs):
        self.r = decay
        self.d = material_con
        self.T = horizon
        #self.decay_mat = (1 - self.r)* np.eye(size_x*size_y+1)
        self.arrival = arrival
        self.lambda_ = lambda_
        self.size = size_x*size_y+2

        
        if arrival == 'geometric':
            self.alpha = kwargs['ALPHA']
            self.trans_mat = self.generate_geometric_trans_mat(size_x,size_y)
            #with np.printoptions(precision=3, suppress=True):
            #    print(self.trans_mat)
            self.compute_nbrhd()
        #if arrival == 'unifrom':
        #    self.S = kwargs['S']
        #    self.trans_mat = self.generate_unifrom_trans_mat(size) 
        #    self.compute_nbrhd()
        return
    
    
    def generate_geometric_trans_mat(self,size_x,size_y):
        '''
        Returns
        -------
        numpy array(sizeXsize)
            time homogeneous transition matrix for a 2DHexGrid system with 
            geometric arrivals
        '''
        n = size_x
        m = size_y

        mapping = {(i,j):i +n*j + 1 for i in range(n) for j in range(m)}
        G = nx.grid_2d_graph(n, m, periodic=False, create_using=None)
        
        for node in list(G.nodes()):
            x = node[0]
            y = node[1]
            
            #add top left
            if ((x-1 ,y-1) in G.nodes()) and ((node,(x-1,y-1)) not in G.edges()):
                G.add_edge(node ,(x-1,y-1))
            
            #add top right
            if ((x+1,y-1) in G.nodes()) and ((node,(x+1,y-1)) not in G.edges()):
                G.add_edge(node ,(x+1,y-1))
        
            #add bottom left
            if ((x-1 ,y+1) in G.nodes()) and ((node,(x-1,y+1)) not in G.edges()):
                G.add_edge(node ,(x-1,y+1))
        
            #add bottom right
            if ((x+1, y+1) in G.nodes()) and ((node,(x+1,y+1)) not in G.edges()):
                G.add_edge(node ,(x+1,y+1))
                
        self.G = nx.relabel_nodes(G, mapping)
        
        input_cells = [1,n,m*n-n+1,m*n]
        escape_cells = [int(m*(n+1)/2)]
        
        #this is just a good time to do this
        self.Restricted = [int(n*m+1),0]
        
        #Create the Shortest path transition matrix
        adjacency_short = np.zeros((n*m+1,n*m+1))
        
        for end in escape_cells:            
            pred = nx.predecessor(self.G, end)
            for cell_from in pred:
                for cell_to in pred[cell_from]:
                    adjacency_short[cell_from -1][cell_to-1] +=1
                
        #Make the escape cells only able to 
        for cell in escape_cells:
            #adjacency_short[cell - 1][cell- 1] = 1
            adjacency_short[cell - 1][n*m] = 1
        
        adjacency_short[n*m][n*m] = 1
            
        #Normalize
        norm = np.sum(adjacency_short,axis=1)
        adjacency_short = adjacency_short /norm.reshape(m*n+1,1)
        
        
        #Create the random walk for the opportinuistic with reflexive edges
        for node in self.G.nodes:
            self.G.add_edge(node,node)
            
        #Normalize to make stochastic
        adjacency_walk = nx.to_numpy_array(self.G)
        
        #Add the reflexive terminal node
        adjacency_walk = np.block([[adjacency_walk,np.zeros((m*n,1))],[np.zeros((1,m*n)),np.array([1.0])]])
        
        #get rid of the outgoing arcs from the escape node that don't go to the terminal node
        for from_cell in escape_cells:
            for to_cell in range(n*m+1):
                if to_cell != int(n*m+1)-1:
                    adjacency_walk[from_cell-1][to_cell] = 0.0
                    
        # make sure the escape node communicates with the terminal node
        for cell in escape_cells:
            adjacency_walk[cell-1][int(n*m+1)-1] = 1
            
        
        norm = np.sum(adjacency_walk,axis=1)
        adjacency_walk = adjacency_walk /norm.reshape(m*n+1,1)

        #Create the transition matrix
        lower_right = (self.lambda_)*adjacency_walk+ (1-self.lambda_)*adjacency_short
        top_left= np.array([1-self.alpha])
        top_right = np.zeros((1,len(adjacency_walk)))
        for cell in input_cells:
            top_right[0][cell-1] = self.alpha/4
        lower_left = np.zeros((len(adjacency_walk),1))

        return  np.block([[top_left,top_right],[lower_left,lower_right]])
    
    
    def sample(self, samp_num, glimpse_func = 'exp', **kwargs):
        '''
        Samples random walks and computes detectable material deposit and 
        glimpse for the sample walks

        Parameters
        ----------
        samp_num : int
            number of smapled walks
        glimpse_func : string, optional
            Choice of glimpse function.
                exp - exponential
                tanh - hyperbolic tangent
            The default is 'exp'.

        Returns
        -------
        None.

        '''
        self.sample = [] # Sample list of walks
        self.samp_num = samp_num
        #Sample the state evolution 
        for j in tqdm(range(samp_num)):
            #initialize the state and sample paramters
            chain = self.make_chain()
            walk = ['0'] # Initial state
            X_state = [self.d if i == 0 else 0 for i in range(self.size)] # Initial state 
            sample_walk = [X_state] # Sample of a single walk
            
            for i in range(1, self.T+1):
                #Get next state
                current_state = walk[-1]
                next_state = chain.next(current_state)
                
                #Deposit and decay the material
                X_state = list(np.dot(np.array(X_state).T,self.r)+ self.d*np.array([self.d if i == int(next_state) else 0 for i in range(self.size)]))
                walk.append(next_state)
                sample_walk.append(X_state)
            self.sample.append(sample_walk)
            
        # generate the glimpse values for every sample    
        self.set_glimpse(glimpse_func, params = {i:kwargs[i] for i in kwargs})
        self.create_time_staged_graph(G = self.G)
        self.calculate_modified_graph(G = self.G) # must happen after create_time_staged_graph
        return

class Bottle_Neck_Prime(Bottle_Neck):
    def sample(self, samp_num, glimpse_func = 'exp', **kwargs):
        '''
        Samples random walks and computes detectable material deposit and 
        glimpse for the sample walks

        Parameters
        ----------
        samp_num : int
            number of smapled walks
        glimpse_func : string, optional
            Choice of glimpse function.
                exp - exponential
                tanh - hyperbolic tangent
            The default is 'exp'.

        Returns
        -------
        None.

        '''
        self.sample = [] # Sample list of walks
        self.samp_num = samp_num
        #Sample the state evolution 
        for j in range(samp_num):
            #initialize the state and sample paramters
            chain = self.make_chain()
            walk = ['0'] # Initial state
            X_state = [self.d if i == 0 else 0 for i in range(self.size)] # Initial state 
            sample_walk = [X_state] # Sample of a single walk
            
            for i in range(1, self.T+1):
                #Get next state
                current_state = walk[-1]
                next_state = chain.next_state(current_state)
                
                #Deposit and decay the material
                X_state = list(np.dot(np.array(X_state).T,self.r)+ self.d*np.array([self.d if i == int(next_state) else 0 for i in range(self.size)]))
                walk.append(next_state)
                sample_walk.append(X_state)
            self.sample.append(sample_walk)
            
        # generate the glimpse values for every sample    
        self.set_glimpse(glimpse_func, params = {i:kwargs[i] for i in kwargs})
        self.create_time_staged_graph()
        self.calculate_modified_graph() # must happen after create_time_staged_graph
        return
class  Two_D_Hex_grid_Prime( Two_D_Hex_grid):
    def sample(self, samp_num, glimpse_func = 'exp', **kwargs):
        '''
        Samples random walks and computes detectable material deposit and 
        glimpse for the sample walks

        Parameters
        ----------
        samp_num : int
            number of smapled walks
        glimpse_func : string, optional
            Choice of glimpse function.
                exp - exponential
                tanh - hyperbolic tangent
            The default is 'exp'.

        Returns
        -------
        None.

        '''
        self.sample = [] # Sample list of walks
        self.samp_num = samp_num
        #Sample the state evolution 
        for j in range(samp_num):
            #initialize the state and sample paramters
            chain = self.make_chain()
            walk = ['0'] # Initial state
            X_state = [self.d if i == 0 else 0 for i in range(self.size)] # Initial state 
            sample_walk = [X_state] # Sample of a single walk
            
            for i in range(1, self.T+1):
                #Get next state
                current_state = walk[-1]
                next_state = chain.next_state(current_state)
                
                #Deposit and decay the material
                X_state = list(np.dot(np.array(X_state).T,self.r)+ self.d*np.array([self.d if i == int(next_state) else 0 for i in range(self.size)]))
                walk.append(next_state)
                sample_walk.append(X_state)
            self.sample.append(sample_walk)
            
        # generate the glimpse values for every sample    
        self.set_glimpse(glimpse_func, params = {i:kwargs[i] for i in kwargs})
        self.create_time_staged_graph(G = self.G)
        self.calculate_modified_graph(G = self.G) # must happen after create_time_staged_graph
        return
