import networkx as nx
import random

class WeightedOpinionDiffusion:
    def __init__(self, graph, epsilon=0.5, bias=0, reduced_weight=True):
        """
        Inizializza il modello di diffusione delle opinioni.

        Parametri:
        - graph (networkx.Graph): Il grafo su cui eseguire la simulazione.
        - epsilon (float): Parametro che indica la threshold che la distanza pesata di opinioni 
                           non deve superare per aggiornare la sua opinione (open mindness).
                           Valore in [0,1] (default: 0.5)
        - bias (float): rappresenta il bias del modello; all'aumentare del valore, 
                        aumenta la probabilità che due opinioni vicine interagiscono 
                        e diminuisce la probabilità di interazione tra due opinioni lontane.
                        Valore in [0,100] (default: 0)
        - reduced_weight (bool): utilizza una versione ridotta dei pesi in modo da avere una minore influenza
                                 nella scelta e nel calcolo dell'opinione (default: True)
        """
        if epsilon < 0 or epsilon > 1: raise ValueError("Il parametro epsilon deve essere un valore compreso in [0,1]")
        if bias < 0 or bias > 100: raise ValueError("Il parametro bias deve essere un valore compreso in [0,100]")
        self.graph = graph
        self.epsilon = epsilon
        self.bias = bias
        self.reduced_weight = reduced_weight
    
    def reduced_weight_function(self, weight_ji):
        """
        Diminuisce l'influenza del peso:
        w_ji=1 => reduced(w_ji)=1,   w_ji=2 => reduced(w_ji)=1.1,   w_ji=3 => reduced(w_ji)=1.2, ...
        
        Parametri:
        - weight_ji (int) Peso dell'arco j -> i, il valore deve essere positivo
        """
        if weight_ji < 0: raise ValueError("Il valore del peso di un arco deve essere positivo")
        return (1 + (weight_ji - 1)/10) 
    
    def weighted_distance(self, opinion_i, opinion_j, weight_ji):
        """
        Calcola la distanza tra le opinioni tenendo conto del peso dell'arco.
        Il peso dell'arco diminuisce la distanza tra opinioni.

        Parametri:
        - opinion_i (float): Opinione del nodo i.
        - opinion_j (float): Opinione del nodo j.
        - weight_ji (int) Peso dell'arco j -> i (arco entrante in i)
        :return: Valore di compatibilità tra 0 e 1.
        """
        if self.reduced_weight:
            return abs(opinion_i - opinion_j)/self.reduced_weight_function(weight_ji)
        else:
            return abs(opinion_i - opinion_j)/weight_ji
         
    def iteration(self):
        """
        Esegue un'iterazione della diffusione delle opinioni sul grafo.
        Essendo il grafico diretto ogni nodo viene aggiornato considerando solo gli archi in entrata.
        Tra i nodi in entrata quello selezionato è estratto con una probabilità:

                p_i(j) = d_ij^(-bias) / sum_(k != i) d_ik^(-bias)

        dove la distanza è pesata con il peso dell'arco.
        Una volta selezionato il nodo l'interazione dell'opinioni consiste in una media pesata:

                o'_i = (o_i + w_ji*o_j)/(w_ji+1)

        dove o'_i è l'opinione aggiornata del nodo i-esimo (il nodo j-esimo non viene aggiornato per via
        della direzione dell'arco).
        L'aggiornamento viene effettuato solo se la distanza pesata d_ji è minore del parametro epsilon   
        """
        nodes_list = list(self.graph.nodes())
        random.shuffle(nodes_list)
        
        new_opinions = {}
        
        for node in nodes_list:
            current_opinion = self.graph.nodes[node].get("opinion", 0.5)
            predecessors = list(self.graph.predecessors(node))
            
            if predecessors:
                # Calcola la probabilità per ogni vicino
                total_prob = 0
                probabilities = {}
                min_dist = 1e-10  # Evita distanze esattamente zero
                for neighbor in predecessors:
                    neighbor_opinion = self.graph.nodes[neighbor].get("opinion", 0.5)
                    edge_weight = self.graph.get_edge_data(neighbor, node, default={"weight": 1})["weight"]
                    dist = self.weighted_distance(current_opinion, neighbor_opinion, edge_weight)
                    dist = max(min_dist, dist) # Assicura che la distanza non sia 0
                    probabilities[neighbor] = dist ** (-self.bias)
                    total_prob += probabilities[neighbor]

                # Normalizza le probabilità
                for neighbor in probabilities:
                    probabilities[neighbor] /= total_prob
                
                # Seleziona un vicino sulla base delle probabilità
                chosen_neighbor = random.choices(predecessors, weights=[probabilities[neighbor] for neighbor in predecessors])[0]
                
                # Calcola la nuova opinione
                neighbor_opinion = self.graph.nodes[chosen_neighbor].get("opinion", 0.5)
                edge_weight = self.graph.get_edge_data(chosen_neighbor, node, default={"weight": 1})["weight"]
                dist = self.weighted_distance(current_opinion, neighbor_opinion, edge_weight)
                
                if dist <= self.epsilon: 
                    if self.reduced_weight: weight = self.reduced_weight_function(edge_weight)
                    else: weight = edge_weight
                    new_opinion = (weight * neighbor_opinion + current_opinion) / (weight + 1)           
                else: 
                    new_opinion = current_opinion
            else:
                new_opinion = current_opinion # Se il nodo non ha archi in ingresso
            
            self.graph.nodes[node]["opinion"] = new_opinion
            new_opinions[node] = new_opinion
        
        return new_opinions


# Esempio di utilizzo
"""
graph = nx.DiGraph()
graph.add_nodes_from([0, 1, 2, 3])
graph.add_edges_from([(0, 1, {"weight": 0.5}), (1, 2, {"weight": 0.7}), (2, 3, {"weight": 0.9})])

# Inizializza le opinioni iniziali per i nodi
for node in graph.nodes():
    graph.nodes[node]["opinion"] = random.uniform(0, 1)

model = WeightedOpinionDiffusion(graph, epsilon=0.5, bias = 1, reduced_weight=True)

# Esegue alcune iterazioni del modello
for _ in range(10):
    new_opinions = model.iteration()
    print(new_opinions)
"""