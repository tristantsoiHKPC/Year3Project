# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import cProfile

#%%

class BaraAlbert():
    def __init__(self, tmax, m):
        self._tinitial = m+1 # tinital needs to be greater than m -> k cannot be < m
        # complete -> unbiased 
        self._G = nx.complete_graph(self._tinitial) 
        self._m = m
        self._tmax = tmax
        self._vertices = list(self._G.nodes)
        self._edges = list(self._G.edges)
        # degree list
        self._PAlist = [vertex for pair in self._edges for vertex in pair]
        self._binCentres = []
        self._degreeList = []

        if self._m > self._tinitial:
            raise ValueError("m cannot exceed number of nodes in initial graph")
            
        if self._tmax <= self._tinitial:
            raise ValueError("tmax has to be greater than tinitial")
    
    def PrefAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._G.add_node(index)
            self._vertices = list(self._G.nodes) # update
            for j in range(self._m):
                randomPA = random.choice(self._PAlist)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while randomPA==index or (randomPA, index) in self._edges:
                    randomPA = random.choice(self._PAlist)
                self._G.add_edge(index,randomPA)
                self._edges = list(self._G.edges)
                self._PAlist.append(index)
                self._PAlist.append(randomPA)
                
            if plot:
                plt.figure()  
                plt.title(f'$t$ = {index+1}')
                nx.draw(self._G, with_labels=True, font_weight='bold')
                plt.show()
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                
    def Optimized_PrefAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._G.add_node(index)
            self._vertices = list(self._G.nodes) # update
            for j in range(self._m):
                randomPA = random.randint(min(self._PAlist),max(self._PAlist)+1)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while randomPA==index or (randomPA, index) in self._edges:
                    randomPA = random.randint(min(self._PAlist),max(self._PAlist)+1)
                self._G.add_edge(index,randomPA)
                self._edges = list(self._G.edges)
                self._PAlist.append(index)
                self._PAlist.append(randomPA)
                
            if plot:
                plt.figure()  
                plt.title(f'$t$ = {index+1}')
                nx.draw(self._G, with_labels=True, font_weight='bold')
                plt.show()
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                
    def RandAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._G.add_node(index)
            self._vertices = list(self._G.nodes) # update
            for j in range(self._m):
                # randomly choose a vertex
                random_choice = random.choice(self._vertices)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while random_choice==index or (random_choice, index) in self._edges:
                    random_choice = random.choice(self._vertices)
                self._G.add_edge(index,random_choice)
                self._edges = list(self._G.edges)
                self._PAlist.append(index)
                self._PAlist.append(random_choice)
                    
            if plot:
                plt.figure()  
                plt.title(f'$t$ = {index+1}')
                nx.draw(self._G, with_labels=True, font_weight='bold')
                plt.show()
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                    
    def Optimized_RandAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._G.add_node(index)
            self._vertices = list(self._G.nodes) # update
            for j in range(self._m):
                # randomly choose a vertex
                random_choice = random.randint(min(self._vertices),max(self._vertices)+1)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while random_choice==index or (random_choice, index) in self._edges:
                    random_choice = random.randint(min(self._vertices),max(self._vertices)+1)
                self._G.add_edge(index,random_choice)
                self._edges = list(self._G.edges)
                self._PAlist.append(index)
                self._PAlist.append(random_choice)
                
            if plot:
                plt.figure()  
                plt.title(f'$t$ = {index+1}')
                nx.draw(self._G, with_labels=True, font_weight='bold')
                plt.show()
                    
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                    
def repeat_method(model_class, N, m, iteration_num, method=""):
    degreeLists = []
    for i in range(iteration_num):
        t=time.time()
        model = model_class(N,m)
        if method == "PrefAttach":
            model.PrefAttach()
        if method == "Optimized_PrefAttach":
            model.Optimized_PrefAttach()     
        if method == "RandAttach":
            model.RandAttach()       
        if method == "Optimized_RandAttach":
            model.Optimized_RandAttach()  
        elapsed_time = time.time() - t
        print(f"repeat_{i} took {elapsed_time} s to run")
        degreeList = model._degreeList
        degreeLists.append(degreeList)
        
    return degreeLists



