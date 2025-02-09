import networkx as nx
import random

class WeightedOpinionDiffusionMean:
    def __init__(self, graph, beta=0.5):
        """
        Inizializza il modello di diffusione delle opinioni.
        
        :param graph: Il grafo su cui eseguire la simulazione.
        :param beta: Parametro che indica la predisposizione al cambiamento (0 <= beta <= 1).
        """
        self.graph = graph
        self.beta = beta
    
    def compatibility(self, opinion_i, opinion_j):
        """
        Calcola il fattore di compatibilità in base alla differenza tra le opinioni.
        
        :param opinion_i: Opinione del nodo i.
        :param opinion_j: Opinione del nodo j.
        :return: Valore di compatibilità tra 0 e 1.
        
            f = 1 - |opinion_i - opinion_j|
        
        Se le opinioni sono uguali (differenza 0) f = 1, mentre se sono agli estremi (differenza 1) f = 0.   
        """
        return 1 - abs(opinion_i - opinion_j)
    
    def iteration(self):
        """
        Esegue un'iterazione della diffusione delle opinioni sul grafo.
        
        Essendo il grafico diretto ogni nodo viene aggiornato considerando solo gli archi in entrata.
        L'aggiornamento viene effetuato tramite il parametro beta:
        
            new_opinion = current_opinion + beta * (aggregated_influence - current_opinion)

        dove aggregated_influence è la media delle opinioni pesata con i pesi degli archi in ingresso
        moltiplicati per la compatibilità tra i due nodi.    

        L'aggiornamento è asincorno, cioè il grafico viene aggiornato ad ogni nodo
        """
        # Lista dei nodi in ordine casuale per aggiornamento asincrono
        nodes_list = list(self.graph.nodes())
        random.shuffle(nodes_list)
        
        new_opinions = {}
        
        for node in nodes_list:
            # Ottiene l'opinione corrente (default 0.5)
            current_opinion = self.graph.nodes[node].get("opinion", 0.5)
            
            influence_numer = 0.0
            influence_denom = 0.0
            
            # Per un grafo diretto, considera i predecessori (solo collegamenti in entrata)
            for neighbor in self.graph.predecessors(node):
                neighbor_opinion = self.graph.nodes[neighbor].get("opinion", 0.5)
                edge_weight = self.graph.get_edge_data(neighbor, node, default={"weight": 1})["weight"]
                
                # Calcola il fattore di compatibilità
                comp = self.compatibility(current_opinion, neighbor_opinion)
                
                influence_numer += edge_weight * comp * neighbor_opinion
                influence_denom += edge_weight * comp
            
            if influence_denom > 0:
                aggregated_influence = influence_numer / influence_denom
                # Aggiornamento differenziale
                new_opinion = current_opinion + self.beta * (aggregated_influence - current_opinion)
                # Clipping per garantire che 0 <= opinion <= 1
                new_opinion = min(max(new_opinion, 0), 1)
            else:
                new_opinion = current_opinion
            
            # Aggiornamento immediato nel grafo (asincrono)
            self.graph.nodes[node]["opinion"] = new_opinion
            new_opinions[node] = new_opinion
        
        return new_opinions


# Esempio utilizzo
"""
# Inizializza il modello
model = WeightedOpinionDiffusion(G)

# (Opzionale) Configurazione dei parametri
config = mc.Configuration()
config.add_model_parameter("beta", 0.5)
model.set_initial_status(config)

# Esegui iterazioni
for t in range(10):
    opinions = model.iteration(t)
    print(f"Iterazione {t+1}: {opinions}")
"""