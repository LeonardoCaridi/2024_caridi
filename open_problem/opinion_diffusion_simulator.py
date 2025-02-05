import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from weighted_opinion_diffusion import WeightedOpinionDiffusion

class OpinionDiffusionSimulator:
    """
    Simula la diffusione delle opinioni su una rete di utenti utilizzando il modello Weighted Opinion Diffusion.
    
    Parametri:
    - graph (networkx.Graph): Grafo della rete sociale con attributi di opinione sui nodi.
    - iterations (int): Numero di iterazioni della simulazione (default: 100).
    - epsilon (float): Parametro di tolleranza per l'aggiornamento delle opinioni (default: 0.5).
    - bias (float): Bias nel processo di aggiornamento delle opinioni (default: 0).
    - random_opinion (bool): Se True, inizializza le opinioni dei nodi casualmente (default: False).
    """
    def __init__(self, graph, iterations=100, epsilon=0.5, bias=0, random_opinion=False):
        self.iterations = iterations
        self.epsilon = epsilon
        self.bias = bias
        self.graph = graph.copy() # Crea una copia per non sovrascirvere i valori di opinion
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)
        self.opinions_matrix = np.zeros((self.num_nodes, self.iterations))
        self.proTrump = []
        self.proBiden = []
        self.neutral = []

        if any('opinion' not in self.graph.nodes[node] for node in self.graph.nodes()):
            self.random_opinion = True  # Forza l'inizializzazione casuale se almeno un nodo non ha un'opinione
        else:
            self.random_opinion = random_opinion
        
        self.model = WeightedOpinionDiffusion(self.graph, epsilon=self.epsilon, bias=self.bias)
        self.initialize_opinions()
    
    def initialize_opinions(self):
        """Inizializza le opinioni dei nodi, casualmente se necessario."""
        if self.random_opinion:
            for node in self.graph.nodes():
                self.graph.nodes[node]['opinion'] = random.uniform(0, 1)
        
        # Memorizza le distribuzioni iniziali delle opinioni
        opinions_array = np.array([self.graph.nodes[node]['opinion'] for node in self.graph.nodes()])
        self.proTrump_init = opinions_array[opinions_array >= 0.666].size
        self.proBiden_init = opinions_array[opinions_array <= 0.333].size
        self.neutral_init = opinions_array[np.logical_and(opinions_array > 0.333, opinions_array < 0.666)].size
    
    def run_simulation(self):
        """Esegue la simulazione per il numero di iterazioni specificato."""
        for t in tqdm(range(self.iterations)):
            new_opinions = self.model.iteration()
            for i, node in enumerate(self.nodes):
                self.opinions_matrix[i, t] = new_opinions[node]
            
            # Aggiorna i conteggi delle opinioni
            opinions_array = np.asarray(list(new_opinions.values()))
            self.proTrump.append(opinions_array[opinions_array >= 0.666].size)
            self.proBiden.append(opinions_array[opinions_array <= 0.333].size)
            self.neutral.append(opinions_array[np.logical_and(opinions_array > 0.333, opinions_array < 0.666)].size)
        
    def plot_opinion_distribution(self, title="Distribuzione opinioni", ax=None, save=False):
        """Genera un istogramma della distribuzione delle opinioni."""
        if ax is None:
            fig, ax = plt.subplots()
        if self.random_opinion: 
            title += " (random init)"
        opinions = np.array([self.graph.nodes[node]['opinion'] for node in self.graph.nodes()])
        counts, bins, patches = ax.hist(opinions, bins=20, edgecolor='black', log=True)
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=0, vmax=1)
        
        for patch, bin_left in zip(patches, bins[:-1]):
            patch.set_facecolor(cmap(norm(bin_left)))
        
        ax.set_xlabel("Opinione")
        ax.set_ylabel("Numero di utenti")
        ax.set_title(title)
        ax.axvline(x=0.333, linestyle="--", color='black', alpha=0.7)
        ax.axvline(x=0.666, linestyle="--", color='black', alpha=0.7)
        if save: plt.savefig(os.path.join("plots", title))
        if ax is None: plt.show()
    
    def plot_opinion_diffusion(self, title="Diffusione opinioni", ax=None, legend=False, save=False):
        """Grafico dell'evoluzione delle opinioni nel tempo."""
        if ax is None:
            fig, ax = plt.subplots()
        if self.random_opinion: 
            title += " (random init)"
        ax.axhline(y=self.proBiden_init, color='b', linestyle='--', alpha=0.7)
        ax.axhline(y=self.proTrump_init, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=self.neutral_init, color='grey', linestyle='--', alpha=0.7)
        
        # Il modello Weighted Opinion Diffusion restituisce i valori dalla prima iterazione in poi,
        # i valori iniziali vanno inseriti manualmente 
        self.proBiden.insert(0,self.proBiden_init)
        self.proTrump.insert(0,self.proTrump_init)
        self.neutral.insert(0,self.neutral_init)
        ax.plot(range(self.iterations+1), self.proBiden, color='b', label='pro-Biden')
        ax.plot(range(self.iterations+1), self.proTrump, color='r', label='pro-Trump')
        ax.plot(range(self.iterations+1), self.neutral, color='grey', label='neutral')
        
        ax.set_xlabel("Iterazioni")
        ax.set_ylabel("Numero di utenti")
        ax.set_title(title)
        if legend: ax.legend()
        if save: plt.savefig(os.path.join("plots", title))
        if ax is None: plt.show()
    
    def plot_opinion_evolution(self, title="Evoluzione opinioni nel tempo", ax=None, save=False):
        """Grafico delle opinioni dei singoli utenti nel tempo."""
        if ax is None:
            fig, ax = plt.subplots()
        if self.random_opinion: 
            title += " (random init)"
        iterations = np.arange(self.iterations)
        first_opinions = self.opinions_matrix[:, 0]
        colors = ["blue" if op <= 0.333 else "red" if op >= 0.666 else "gray" for op in first_opinions]
        
        for i in range(self.num_nodes):
            ax.plot(iterations, self.opinions_matrix[i, :], color=colors[i], alpha=0.5, linewidth=0.5)
        
        ax.set_xlabel("Iterazioni")
        ax.set_ylabel("Opinione")
        ax.set_title(title)
        if save: plt.savefig(os.path.join("plots", title))
        if ax is None: plt.show()

            
            

# Esempio di utilizzo:
# simulator = OpinionDiffusionSimulator(G, iterations=100, epsilon=0.5, bias=1, random_opinion=False)
# simulator.run_simulation()
# simulator.plot_opinion_distribution()
# simulator.plot_opinion_diffusion()
# simulator.plot_opinion_evolution()
