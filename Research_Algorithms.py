# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:18:04 2023

@author: Joseph Kennedy
"""

import numpy as np
from tqdm import tqdm
import networkx as nx
import progressbar
import time




class ACO():
    def __init__(self,AOI,objective,heuristic = None, pop_sz = 10, stop_it = 1000,rho = 0.9,alpha = 1, beta = 1, plot = False, get_hist = False):
        

        self.heuristic = heuristic
        self.eval_obj = objective
        
        self.Restricted = AOI.Restricted
        self.sol_sz = AOI.T
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        
        self.pd_terms = [0 for i in range(self.sol_sz)]
        self.plot = plot
        self.get_hist = get_hist
        self.stop_mark = stop_it
        self.nbrhd = AOI.nbrhd
        self.pop_sz = pop_sz
        self.sz = AOI.size
        self.init_state_action()
        self.eta_heuristic()
        return
         
    def init_state_action(self):
        '''
        Initializes the state action table
        State represents the keys is (cell, time)
        Actions represent the neighbors and their accumulated pheromone
        '''
        self.tau = {}
        self.eta = {}
        

        for i in self.nbrhd: # loop for cell
            for j in range(self.sol_sz+1): #loop for time
                nhbrs = self.nbrhd[i]
                nhbrs_pher = {}
                for k in nhbrs:

                    nhbrs_pher[k] = 1/len(nhbrs)
                if j != self.sol_sz:
                    self.tau[(i,j)] = nhbrs_pher
                    self.eta[(i,j)] = nhbrs_pher.copy()
                else:
                    self.tau[(i,j)] = {'t':1}
                    self.eta[(i,j)] = {'t':1}
                    
        #States for start s and t that can transfer to anywhere accept the resptricted nodes
        start_to_nodes = {}
        for i in range(1,self.sz):
            start_to_nodes[i] = 1/(self.sz-len(self.Restricted))
            
        #cannot start at a restricted node
        for i in self.Restricted:
            start_to_nodes[i] = 0
        self.tau[('s',-1)] = start_to_nodes
        self.eta[('s',-1)] = start_to_nodes.copy()
        return
    
    
    def rand_norm(self,nbrs):
        '''
        Takes a dictionary of neighbor moves, normalizes the weights and returns a distribution of next choice
        '''
        return list(np.array(list(nbrs.values()))/sum(i for i in nbrs.values()))
    
    def eta_heuristic(self):
        # calculate the probability of detect in next cell
        for state_action in self.eta:
            for next_state in self.eta[state_action]:
                if next_state not in self.Restricted:
                    self.eta[state_action][next_state] = self.heuristic(state_action[1]+1,next_state)
                else:
                    self.eta[state_action][next_state] = 0

        #Make sure we don;t reward movemetns to restricted cells
        for i in self.eta[('s', -1)]:
            if i in self.Restricted:
                self.eta[('s', -1)][i] = 0
            else:
                self.eta[('s',-1)][i] = 1/(self.sz-len(self.Restricted))
        return
    
    def create_sol(self, uniform = False):
        '''
        Uses the pheremones to randomly sample next moves to create an optimal solution
        '''

        # Initialize the random walk from s
        sol = ['s']
        
        # Generate the rest of the walk 
        for i in range(-1, self.sol_sz-1):
            
            # Get random neighbors of last node
            vals_tau = self.tau[(sol[-1],i)]
            vals_eta = self.eta[(sol[-1],i)]
            
            # Combine the taus and eta`s (no need to normalize here, we only normalize when making the walk)
            vals = {j:vals_tau[j]**self.alpha +vals_eta[j]**self.beta for j in vals_tau}
            # drop the restricted moves
            for i in self.Restricted:
                vals.pop(i,'')
            # normalize 
            dist = self.rand_norm(vals)
            
            # sample next cell to search
            sol.extend(list(np.random.choice(list(vals.keys()),p =dist,size = 1))) 

        return sol, self.eval_obj(sol)
    
    
    def decay_pher(self):
        '''
        Decay the existing populations pheromones before depositing
        '''
        for cell_it in self.tau:
            for cell_jt in self.tau[cell_it]:
                self.tau[cell_it][cell_jt] = self.rho*self.tau[cell_it][cell_jt]
        return
    
    def deposit_pher(self,sol, vals,itt):
        '''
        Deposits pheromones onto states based on performance of the path overall
        '''
        #Calculate the Taus
        for t_section in range(-1,self.sol_sz):
            try:
                tau = self.tau[(sol[t_section+1],t_section)][sol[t_section+2]]
                remaining_prob = np.sum([vals[j] for j in range(1,t_section - 1)]) #the 1/pop_sz averages over the population
                self.tau[(sol[t_section+1],t_section)][sol[t_section+2]] = tau + np.sum([vals[j] for j in range(t_section,self.sol_sz)])/((self.pop_sz)*(1-remaining_prob))
            except:
                pass
        return
    
    
    def argmax(dic):
        '''
        Returns the Key corrresponding to the largest value of a dictionary
        '''
        max_value = max(dic.values())  # maximum value
        return [k for k, v in dic.items() if v == max_value][0]
    
    
    def comput_best_from_greedy(self):
        #compute the greedy choice from state action table
        greedy_sol = ['s']
        for t in range(-1,self.sol_sz):
            greedy_sol.extend([self.argmax(self.tau[(greedy_sol,t)])])
        if self.eval_obj(greedy_sol[1:])[0]>self.best_obj:
            return 1, greedy_sol
        else:
            return -1, greedy_sol
    
    def ACO(self):
        '''
        Execute the ACO algorithm and record any necesary algorithmic hisotry for analysis
        '''
        #Set initial solution temp and best
        best_obj = 0
        best_sol = []
        
        # Set outputs for analysis
        Best_History_Val = []
        Best_History_Sol = []
            
        itr = 0
        num_itt = self.stop_mark
        update = False
        
        for j in tqdm(range(num_itt)): 
            
            sol_lst = []
            obj_val_lst = []
            
            for k in range(self.pop_sz):
                # Get new solution
                sol, obj_val = self.create_sol()
                    
                sol_lst.extend([sol])
                obj_val_lst.extend([obj_val])
                
                # Best so Far?
                if obj_val[0] > best_obj:
                    best_obj = obj_val[0]
                    best_sol = sol
                    update = True
     
            # Update Pheromones
            self.decay_pher()
            for k in range(self.pop_sz):
                self.deposit_pher(sol_lst[k],obj_val_lst[k][1],itr)

            # Update the history
            if self.get_hist:
                Best_History_Val.extend([best_obj])
                if update:
                    Best_History_Sol.extend([best_sol])
                    update = False

        return best_obj, best_sol, Best_History_Val, Best_History_Sol
 
class ACO_Prime(ACO):
    def ACO(self):
        '''
        Execute the ACO algorithm and record any necesary algorithmic hisotry for analysis
        '''
        #Set initial solution temp and best
        best_obj = 0
        best_sol = []
        
        # Set outputs for analysis
        Best_History_Val = []
        Best_History_Sol = []
            
        itr = 0
        num_itt = self.stop_mark
        update = False
        
        for j in range(num_itt): 
            
            sol_lst = []
            obj_val_lst = []
            
            for k in range(self.pop_sz):
                # Get new solution
                sol, obj_val = self.create_sol()
                    
                sol_lst.extend([sol])
                obj_val_lst.extend([obj_val])
                
                # Best so Far?
                if obj_val[0] > best_obj:
                    best_obj = obj_val[0]
                    best_sol = sol
                    update = True
     
            # Update Pheromones
            self.decay_pher()
            for k in range(self.pop_sz):
                self.deposit_pher(sol_lst[k],obj_val_lst[k][1],itr)

            # Update the history
            if self.get_hist:
                Best_History_Val.extend([best_obj])
                if update:
                    Best_History_Sol.extend([best_sol])
                    update = False

        return best_obj, best_sol, Best_History_Val, Best_History_Sol
    def create_sol(self, uniform = False):
        '''
        Uses the pheremones to randomly sample next moves to create an optimal solution
        '''

        # Initialize the random walk from s
        sol = ['s']
        
        # Generate the rest of the walk 
        for i in range(-1, self.sol_sz-1):
            
            # Get random neighbors of last node
            vals_tau = self.tau[(sol[-1],i)]
            vals_eta = self.eta[(sol[-1],i)]
            
            # Combine the taus and eta`s (no need to normalize here, we only normalize when making the walk)
            vals = {j:vals_tau[j]**self.alpha +vals_eta[j]**self.beta for j in vals_tau}
            # drop the restricted moves
            for i in self.Restricted:
                vals.pop(i,'')
            # normalize 
            dist = self.rand_norm(vals)
            
            # sample next cell to search
            sol.extend(list(np.random.choice(list(vals.keys()),p =dist,size = 1))) 

        return sol, self.eval_obj(sol)  
class random_search(ACO):
    def RAND(self, get_all_path = False):
        '''
        Execute the ACO algorithm and record any necesary algorithmic hisotry for analysis
        '''
        #Set initial solution temp and best
        best_obj = 0
        best_sol = []
        
        # Set outputs for analysis
        Best_History_Val = []
        Best_History_Sol = []
            
        #itr = 0
        num_itt = self.stop_mark
        update = False
        
        for j in tqdm(range(num_itt)): 
            
            sol_lst = []
            obj_val_lst = []
            
            for k in range(self.pop_sz):
                # Get new solution
                sol, obj_val = self.create_sol()
                    
                sol_lst.extend([sol])
                obj_val_lst.extend([obj_val])
                
                # Best so Far?
                if obj_val[0] > best_obj:
                    best_obj = obj_val[0]
                    best_sol = sol
                    update = True
     
            # Update Pheromones
            #self.decay_pher()
            #for k in range(self.pop_sz):
            #    self.deposit_pher(sol_lst[k],obj_val_lst[k][1],itr)

            # Update the history
            if self.get_hist:
                Best_History_Val.extend([best_obj])
                if update:
                    Best_History_Sol.extend([best_sol])
                    update = False

        return best_obj, best_sol, Best_History_Val, Best_History_Sol
    def create_sol(self, uniform = False):
        '''
        Uses the pheremones to randomly sample next moves to create an optimal solution
        '''

        # Initialize the random walk from s
        sol = ['s']
        
        # Generate the rest of the walk 
        for i in range(-1, self.sol_sz-1):
            
            # Get random neighbors of last node
            vals_tau = self.tau[(sol[-1],i)]
            #vals_eta = self.eta[(sol[-1],i)]
            
            # Combine the taus and eta`s (no need to normalize here, we only normalize when making the walk)
            vals = {j:vals_tau[j]**self.alpha  for j in vals_tau}
            # drop the restricted moves
            for i in self.Restricted:
                vals.pop(i,'')
            # normalize 
            dist = self.rand_norm(vals)
            
            # sample next cell to search
            sol.extend(list(np.random.choice(list(vals.keys()),p =dist,size = 1))) 

        return sol, self.eval_obj(sol)      
 
    
 
class Stupid_Search(ACO):
    def __init__(self,AOI,objective):
        return
    def Search(self, sample_size):
        return

class Branch_and_Bound_Static():
    
    def __init__(self, AOI,objective, bound = None, bound_type = 0, time_out = np.inf, max_nodes = np.inf):
        
        if bound_type == 0:
            self.Graph = AOI.Staged_Graph
        elif bound_type == 2:
            self.Graph = AOI.Staged_Graph_Modified
        self.Restricted = AOI.Restricted
        self.objective = objective
        self.T = AOI.T
        self.time_out = time_out
        self.timed_out = False
        self.max_nodes = max_nodes
        self.gap_record = []
        self.p_hat_record = []
        return
            
    def _pop_best(self,t):
        _max = 0
        _argmax = None
        
        # Find the best
        for i in range(len(self.K[t])):
            if self.K[t][i]['p_bar'] > _max:
                _max = self.K[t][i]['p_bar']
                _argmax = i
        
        #Remove best from K
        #pop = self.K[t].pop(i)
        pop = self.K[t].pop(_argmax)
        return pop
    
    def _push(self,x):
        return
    
    def p_upper(self,node,t):
        '''
        Returns the upper bound on the probability of detect for the best path from the current stage to the END stage
        '''

        if node == 'start':
            return -1*nx.dijkstra_path_length(self.Graph,'start',('END',self.T+1))
        else:
            return -1*nx.dijkstra_path_length(self.Graph,(node,t),('END',self.T+1))
        
    def p_at(self,path):
        '''
        Returns the cummulative probability of a path
        '''
        return self.objective(path)#[1:]
    
    #def p_up_to(self,hist,j):
    #    return
    
    def extend(self,current, t, track_performance, verbose):
        '''
        Parameters
        ----------
        current : Dictionary
            Current node to be extended.
        t : int
            current length of path.
        track_performance : Bool
            Control on whether or not to track statistics.
        verbose : bool
            control on whether or not to print each node being branched.

        Returns
        -------
        None.

        '''
        #print(nx.edges(self.Graph))
        if current['path'][-1] == 'start':
            neigh = nx.neighbors(self.Graph,current['path'][-1])
            #print('This is the list of neigh in the extend', list(neigh))
        else:
            neigh = nx.neighbors(self.Graph,(current['path'][-1],t))
        #print(list(neigh))
        for j in list(neigh):
            # calculate the bound to go and P_bar for that extension
            j_path = current['path']+[j[0]]
            p_at_j = self.p_at(j_path)
            
            new = {'path':j_path,
                   'time':t+1,
                   'p_to_j': current['p_to_j'] + p_at_j,
                   'p_upper':self.p_upper(j[0],j[1])}

            new['p_bar'] = new['p_to_j']  + new['p_upper'] 

                
            if track_performance: #update stats
                self.statistics['Total Nodes'] += 1
                
            if len(new['path']) == self.T +1:
                if track_performance:
                    self.statistics['Full Nodes'] += 1
                
                if new['p_to_j'] >= self.p_hat:
                    if verbose:
                        print('New Best ',new['path'])
                    self.p_hat = new['p_to_j']
                    self.best_path = new['path']
                    self.p_hat_record.append(self.p_hat)
                    
            elif track_performance: #update stats
                self.statistics['Subpath Nodes'] += 1
                
            #push the triplet into K(t+1)
            if t+1 not in self.K:
                self.K[t+1] = []
            self.K[t+1].extend([new])
        return
    
    
    def BnB_Procedure(self, starting_bound = 0,verbose = False, track_performance = True):
        '''
        Parameters
        ----------
        starting_bound : float in [0,1]
            A good guess on the optimal probability of detection. 
            The default is 0.
        verbose : bool, optional
            When true every surrent node being examine will be printed.
            The default is False.

        Returns
        -------
        p_hat : float
            Best probability of detect found by the optimal path .
        best_path : list
            Path corresponding to best probability of detection.

        '''
        # These are used for tracking performance
        if track_performance:
            self.length_at_fathoming = {(i+1):0 for i in range(self.T)} # this is for comparison
            self.statistics = {'Total Nodes': 1,
                               'Full Nodes': 0,
                               'Subpath Nodes': 1,
                               'Fathomed Nodes': 0,
                               'gap': np.inf}
        t = 0
        self.p_hat = starting_bound #current best objective
        self.best_path = []
        self.K = {0: [{'path':['start'], #current path
                       'time': 0,
                       'p_to_j':0, #probability up to the last node
                       'p_upper':1}]} #upper bound on the probabilityself.p_upper('start',0)
        self.K[0][0]['p_bar'] = self.K[0][0]['p_to_j']  + self.K[0][0]['p_upper']
        
        
        #The while loop makes using tqdm hard so we use this progress bar
        self.min_upper = 1 # self.p_upper('start',0)
        bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=100)
        bar.start()
        #pbar = tqdm(total = 100)
        start = time.time()
        

        while  (time.time() - start < self.time_out) and (self.statistics['Full Nodes'] < self.max_nodes):
            while True:
                #print(self.K)
                #If K(t) is empty replace t with t-1:
                if not self.K[t]:
                    t-=1
                    
                    #if t = 0 stop BnB_Procedure
                    if (not t):# or (self.gap_record[-1] == 0):
                        self.time_out = False
                        if track_performance:
                            #self.statistics['gap'] = self.gap_record[-1]#self.min_upper - self.p_hat
                            self.statistics['run time'] = time.time() - start
                        return self.p_hat, self.best_path
                    #else do nothing

                #else pop the triplet with largest P_bar from K(t) and break out of this loop
                else:
                    current = self._pop_best(t)
                    if verbose:
                        print('current',current['path'], 'current[p_to_j]', current['p_to_j'], 'current[p_upper]',current['p_upper'])
                    break
                    
            #if The P_bar that was popped out is at most p_hat then phathom this path
            #print(self.p_hat)
            if  current['p_bar'] < self.p_hat:#current['p_to_j'] + current['p_upper']
                
                if track_performance and current['time'] < self.T: #update stats
                    self.length_at_fathoming[current['time']] += 1
                    self.statistics['Fathomed Nodes'] += 1
                
                if verbose:
                    print('Path : ', current['path'],' is fathomed',current['p_to_j'],'+',current['p_upper'],'<=',self.p_hat)
                #print('min_upper', self.min_upper)
                #if self.min_upper >  current['p_bar']:
                #    self.min_upper = current['p_bar']
                #    self.gap_record.append(self.min_upper - self.p_hat)
                                   
            else:
                #print('min_upper', self.min_upper)
                if self.min_upper >  current['p_bar']:
                    self.min_upper = current['p_bar']
                    #self.gap_record.append(self.min_upper - self.p_hat)
                    #if self.gap_record[-1] == 0.0:
                    #    self.statistics['run time'] = time.time() - start
                    #    return self.p_hat, self.best_path
                        
                    
                #if self.min_upper - self.p_hat == 0.0:
                #    return self.p_hat, self.best
                    
                #if t<T
                if t < self.T:
                    # extended from node and create the new triplets
                    #print('I am extending')
                    self.extend(current,
                                t,
                                track_performance = track_performance,
                                verbose = verbose)
                    t+=1
                
            
            #update gap and progress bar
            gap = int(100*(1 - (min([self.min_upper,1]) - self.p_hat)/min([self.min_upper,1])))
            #if track_performance:
            #    try:
            #        self.statistics['gap'] = self.gap_record[-1]#self.min_upper - self.p_hat
            #    except:
            #        self.statistics['gap'] = self.min_upper - self.p_hat
            
            try:
                #pbar.update(1)
                bar.update(gap)
            except:
                pass
            time.time() - start
        # Finish the progress bar update
        bar.update(100)
        bar.finish()
        #pbar.close()
        if time.time() - start > self.time_out:
            self.timed_out = True
            if track_performance:
                self.statistics['run time'] = time.time() - start
        return self.p_hat, self.best_path

class Enumerate_All_Walks(Branch_and_Bound_Static):
    def __init__():
        return
    
    def BnB_Procedure(self, starting_bound = 0,verbose = False, track_performance = True):
            '''
            Parameters
            ----------
            starting_bound : float in [0,1]
                A good guess on the optimal probability of detection. 
                The default is 0.
            verbose : bool, optional
                When true every surrent node being examine will be printed.
                The default is False.
    
            Returns
            -------
            p_hat : float
                Best probability of detect found by the optimal path .
            best_path : list
                Path corresponding to best probability of detection.
    
            '''
            # These are used for tracking performance
            if track_performance:
                self.length_at_fathoming = {(i+1):0 for i in range(self.T)} # this is for comparison
                self.statistics = {'Total Nodes': 1,
                                   'Full Nodes': 0,
                                   'Subpath Nodes': 1,
                                   'Fathomed Nodes': 0 }
            t = 0
            self.p_hat = starting_bound #current best objective
            self.best_path = []
            self.K = {0: [{'path':['start'], #current path
                           'time': 0,
                           'p_to_j':0, #probability up to the last node
                           'p_upper':1}]} #upper bound on the probabilityself.p_upper('start',0)
            self.K[0][0]['p_bar'] = self.K[0][0]['p_to_j']  + self.K[0][0]['p_upper']
            
            
            #The while loop makes using tqdm hard so we use this progress bar
            self.min_upper = 1 # self.p_upper('start',0)
            bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], maxval=100)
            bar.start()
            #pbar = tqdm(total = 100)
            while True:
                while True:
                    #If K(t) is empty replace t with t-1:
                    if not self.K[t]:
                        t-=1
                        
                        #if t = 0 stop BnB_Procedure
                        if not t:
                            return self.p_hat, self.best_path
                        #else do nothing
                        
                    #else pop the triplet with largest P_bar from K(t) and break out of this loop
                    else:
                        current = self._pop_best(t)
                        if verbose:
                            print('current',current['path'], 'current[p_to_j]', current['p_to_j'])
                        break
                        
                #if The P_bar that was popped out is at most p_hat then phathom this path
                if  current['p_to_j'] + current['p_upper'] <= self.p_hat:#current['p_to_j'] + current['p_upper']
                    
                    if track_performance and current['time'] < self.T: #update stats
                        self.length_at_fathoming[current['time']] += 1
                        self.statistics['Fathomed Nodes'] += 1
                    
                    if verbose:
                        print('Path : ', current['path'],' is fathomed',current['p_to_j'],'+',current['p_upper'],'<=',self.p_hat)
                   
                else:
                    if self.min_upper >  current['p_bar']:
                        self.min_upper = current['p_bar']
                    #if t<T
                    if t < self.T:
                        # extended from node and create the new triplets 
                        self.extend(current,
                                    t,
                                    track_performance = track_performance,
                                    verbose = verbose)
                        t+=1
                    
    
                #update gap and progress bar
                gap = int(100*(1 - (min([self.min_upper,1]) - self.p_hat)/min([self.min_upper,1])))
                
                try:
                    #pbar.update(1)
                    bar.update(gap)
                except:
                    pass
                
            # Finish the progress bar update
            bar.update(100)
            bar.finish()
            #pbar.close()
            return self.p_hat, self.best_path
    
    
class Branch_and_Bound_Dynamic(Branch_and_Bound_Static):
    def __init__(self, AOI,objective, time_out = np.inf, bound = None, max_nodes = np.inf):

        self.Graph = AOI.Staged_Graph
        self.Restricted = AOI.Restricted
        self.objective = objective
        self.T = AOI.T
        self.samp_num = AOI.samp_num
        self.glimpse_array = AOI.glimpse_array
        self.time_out = time_out
        self.max_nodes = max_nodes
        self.timed_out = False
        return
    

    def p_upper(self, path,node,  time):
        '''
        Returns the upper bound on the probability of detect for the best path
        from the current stage to the END stage given the failed detects prior 
        '''
        weights = nx.get_edge_attributes(self.Graph, 'weight')
        
        # Reformat the time-staged graph weight
        if (node == 'start')or (len(path) <= 2):
            self.Graph_modified = self.Graph
            
        else:
            
            #Calculate the modified edge weights
            for edge in self.Graph.edges():
                if edge[0] != 'start':
                    if edge[0][1] > time:
                        t = len(path[1:])
                        pd = 0
                        for n in range(self.samp_num):
                            
                            # calculate the probability of failure
                            try:
                                prod = self.glimpse_array[n][edge[1][1]][edge[1][0]]
                            except:
                                break
                            for u in range(1,t+1):
                                prod *= 1-self.glimpse_array[n][u][path[u]]
                                
                            #Calculate the posterior probability of detect in z[t] at time t given sample path n
                            try:
                                pd += (1/self.samp_num)*prod
                            except:
                                pass
                            
                        weights[edge] = {'weight':-1*pd} 
                    else:
                        weights[edge] = {'weight':weights[edge]}
                else:
                    weights[edge] = {'weight':weights[edge]}
            
            #create the graph
            self.Graph_modified = self.Graph.copy()
            nx.set_edge_attributes(self.Graph_modified, weights)
        if node == 'start':
            return -1*nx.dijkstra_path_length(self.Graph_modified,'start',('END',self.T+1))
        
        else:
            return -1*nx.dijkstra_path_length(self.Graph_modified,(node,time),('END',self.T+1))
        
        
    def extend(self, current, t, track_performance, verbose):
        '''
        Parameters
        ----------
        current : Dictionary
            Current node to be extended.
        t : int
            current length of path.
        track_performance : Bool
            Control on whether or not to track statistics.
        verbose : bool
            control on whether or not to print each node being branched.

        Returns
        -------
        None.

        '''
        if current['path'][-1] == 'start':
            neigh = nx.neighbors(self.Graph,current['path'][-1])
        else:
            neigh = nx.neighbors(self.Graph,(current['path'][-1],t))
        
        for j in list(neigh):
            # calculate the bound to go and P_bar for that extension
            j_path = current['path']+[j[0]]
            p_at_j = self.p_at(j_path)
            
            new = {'path':j_path,
                   'time':t+1,
                   'p_to_j': current['p_to_j'] + p_at_j,
                   'p_upper':self.p_upper(path = j_path, node = j[0], time = j[1])}
            
            new['p_bar'] = new['p_to_j']  + new['p_upper'] 
            
            if track_performance: #update stats
                self.statistics['Total Nodes'] += 1
                
            if len(new['path']) == self.T +1:
                if track_performance:
                    self.statistics['Full Nodes'] += 1
                
                if new['p_to_j'] >= self.p_hat:
                    if verbose:
                        print('New Best ',new['path'])
                    self.p_hat = new['p_to_j']
                    self.best_path = new['path']
                    
            elif track_performance: #update stats
                self.statistics['Subpath Nodes'] += 1
                
            #push the triplet into K(t+1)
            if t+1 not in self.K:
                self.K[t+1] = []
            self.K[t+1].extend([new])
        return
    
def DSST(p,alpha,T):
    
    '''
    Computes optimal distribution of resource quantity T across cells
    @input: - p: dictionary of cell names as keys and probabilities as values
            - alpha: dictionary of cell names and coresponding search parameter 
                    Tradiationally alpha_i = W_i*v_/A_i where
                    W_i is the sweepwidth
                    v_i is the velocity 
                    A_i is the area of the cell
                    
    @output: Dictionary of cell names and cooresponding allocation
    '''
    num_cells = len(p)
    
    # sort the indexes
    temp = {i:p[i]*alpha[i] for i in p}
    sorted_index = {k:v  for k, v in sorted(temp.items(),reverse = True, key=lambda item: item[1])}
    index_order = {i:list(sorted_index.keys())[i] for i in range(num_cells)}
    p = [p[index_order[i]] for i in index_order] # reorder p
    alpha = [alpha[index_order[i]] for i in index_order] # reorder alpha
    alpha_p = [p[i]*alpha[i] for i in range(num_cells)] # Values that determe the order 
    
    #Calculate the y's
    y = []
    for i in range(num_cells-1):
        y.append(np.log(alpha_p[i]/alpha_p[i+1]))

    #Salculate the S's from the y's
    S = [0]
    for i in range(1,num_cells):
        #print(i,[j for j in range(i)], [y[j:i] for j in range(i)])
        S.append(sum([(1/alpha[j])*sum(y[j:i]) for j in range(i)]))

    # Determine the allocation case
    for i in range(1,num_cells):
        if (S[i-1] < T) and (T <= S[i]):
            case = True
            j_star = i
            a = (T-S[j_star-1])/(S[j_star]-S[j_star - 1])
            
            break
        else:
            case = False
            a = (T-S[num_cells-1])/(sum([1/alpha[j] for j in range(num_cells)]))
    
    # Make the allocations
    f_star = []
    if case:
        for j in range(num_cells):
            if j <= j_star-1:
                #sum([y[k] for k in range(j, j_star)])
                #y[j_star]
                f_star.append((1/alpha[j])*sum(y[j:j_star-1]) + a*y[j_star-1]/alpha[j])
            else:
                f_star.append(0)
    else:
        for j in range(num_cells):
            f_star.append((1/alpha[j])*sum(y[j:]) + a/alpha[j])
    
    normalizing_const = sum(p[i]*(1- np.exp(alpha[i]*f_star[i])) for i in range(num_cells))
    posterior = {}
    for i in range(num_cells):
        posterior[index_order[i]] = (1/normalizing_const)*p[i]*(1- np.exp(alpha[i]*f_star[i]))
        
    return {index_order[i]:f_star[i] for i in range(num_cells)}, posterior

class Myopic_Search():
    def __init__(self,AOI,objective):
        self.eval_obj = objective
        
        self.Restricted = AOI.Restricted
        self.sol_sz = AOI.T
        self.pd_terms = [0 for i in range(self.sol_sz)]
        self.nbrhd = AOI.nbrhd
        self.sz = AOI.size
        self.Graph = AOI.Staged_Graph
        return
    

    def run(self):
            
        # Initialize search
        self.sol = ['start']
        t = 0
        print("Initializing search...")  # Print statement
        mx = 0
        
        #For first search start in a entry node
        first = True
        while True:
            
            # Calculate the best place to search among all neighbors
            if self.sol[-1] == 'start':    
                neigh = nx.neighbors(self.Graph,self.sol[-1])
                #print('I am at the start')
            else:
                neigh = nx.neighbors(self.Graph,(self.sol[-1],t))
                #print('I am not at the start')
            
            nbr_lst = list(neigh)
            # If first start in most likely and 1 should always be a entry node
            if first:
                self.sol = self.sol + [nbr_lst[0][0]]
                first = False
                mx = self.eval_obj(self.sol[1:])[0]
            else:
                #print(f"Current node: {self.sol[-1]}, Neighbors: {list(neigh)}")  # Print statement
                #print(nbr_lst)
                #print(nbr_lst)
                if nbr_lst[-1][0] == 'END':
                    break
                else:
                    for node in nbr_lst:
                        #print('node',node)
                        temp = self.sol + [node[0]]
                        obj = self.eval_obj(temp[1:])
                        #print(self.sol[-1], node[0], obj[0]  )
                        #print('obj',obj)
                        if obj[0] >= mx:
                            
                            best_nbr = node[0]
                            #print('update best_nbr',best_nbr)
                            mx = obj[0]
                    
                    # Extend the solution to one of the neighbors
                    self.sol = self.sol + [best_nbr]
                    #print(f"Extending solution to the neighbor {best_nbr}, current solution: {self.sol}")  # Print statement
                    t +=1
                # except:
                #     print("No neighbors found, breaking loop...")  # Print statement
                #     break
            # Move 
        return self.sol, mx
    
            
    
