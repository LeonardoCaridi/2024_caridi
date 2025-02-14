import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from cdlib import algorithms, TemporalClustering
import os
from tqdm import tqdm
from collections import defaultdict

class CommunityEvolutionAnalysis:
    """
    Classe per l'analisi dell'evoluzione delle comunità nel tempo a partire da snapshot
    di grafi.
    """
    def __init__(self, snapshot_graphs, threshold_percentile=75):
        """
        Parametri:
        - snapshot_graphs (list of networkx.Graph): lista di grafi (uno per snapshot)
        - threshold_percentile (int) : percentile usato per filtrare i matching (default 75)
        """
        self.snapshot_graphs = snapshot_graphs
        self.threshold_percentile = threshold_percentile
        self.tc = TemporalClustering()  # Istanza per gestire le clusterizzazioni temporali
        self.matches = None            # Lista di matching (tuple)
        self.matches_array = None      # Array numpy dei matching
        self.filtered_matches = None   # Matching filtrati in base al threshold
        self.evolution_graph = None    # Grafo evolutivo (NetworkX DiGraph)

    def compute_clusterings(self):
        """Calcola e salva le clusterizzazioni per ciascun snapshot."""
        for t, graph in enumerate(self.snapshot_graphs):
            # Algoritmo di clustering (ad es. Louvain, può essere modificato)
            communities = algorithms.louvain(graph)
            self.tc.add_clustering(communities, t)

    @staticmethod
    def jaccard_similarity(comm1, comm2):
        """
        Calcola la similarità di Jaccard tra due comunità.

        Parametri:
        - comm1: iterabile di elementi della comunità 1
        - comm2: iterabile di elementi della comunità 2
        
        :return: valore di similarità
        """
        set1, set2 = set(comm1), set(comm2)
        union = set1 | set2
        if len(union) == 0:
            return 0
        return len(set1 & set2) / len(union)

    def compute_matches(self):
        """
        Per ciascuna coppia di snapshot consecutivi, individua per ogni comunità
        il match migliore in base alla Jaccard similarity.
        """
        matches = []
        n_snapshots = len(self.snapshot_graphs)
        for t in range(n_snapshots - 1):
            clustering_t = self.tc.clusterings[t]      # Comunità al tempo t
            clustering_t1 = self.tc.clusterings[t + 1]   # Comunità al tempo t+1

            # Per ogni comunità in t, trova il migliore match in t+1
            for i, com1 in enumerate(clustering_t.communities):
                best_jaccard = 0
                best_match = None

                for j, com2 in enumerate(clustering_t1.communities):
                    score = self.jaccard_similarity(com1, com2)
                    if score > best_jaccard:
                        best_jaccard = score
                        best_match = (f"{t}_{i}", f"{t+1}_{j}", score)

                if best_match is not None:
                    matches.append(best_match)

        self.matches = matches
        self.matches_array = np.array(matches)

    def filter_matches_by_threshold(self):
        """
        Filtra i matching utilizzando il percentile specificato sui punteggi Jaccard.
        
        :return: valore della soglia usata
        """
        if self.matches is None:
            raise ValueError("Eseguire prima compute_matches().")
        jaccard_scores = np.array([m[2] for m in self.matches])
        threshold = np.percentile(jaccard_scores, self.threshold_percentile)
        self.filtered_matches = self.matches_array[jaccard_scores >= threshold]
        print(f"Matching filtrati con soglia (percentile {self.threshold_percentile}): {threshold:.3f}")
        return threshold

    def plot_jaccard_histogram(self):
        """
        Crea e mostra un istogramma dei punteggi Jaccard con la soglia evidenziata.
        """
        if self.matches is None:
            raise ValueError("Eseguire prima compute_matches().")
        jaccard_scores = np.array([m[2] for m in self.matches])
        threshold = np.percentile(jaccard_scores, self.threshold_percentile)
        
        plt.hist(jaccard_scores, log=True, edgecolor='black')
        plt.axvline(x=threshold, linestyle='--', color='r', label=f'Threshold ({threshold:.3f})')
        plt.title("Istogramma dei punteggi Jaccard")
        plt.xlabel("Jaccard score")
        plt.ylabel("Numero di matching")
        plt.legend()
        plt.show()

    def community_evolution_graph(self, matches_array):
        """
        Costruisce un grafo evolutivo direzionale a partire dai matching.
        Per ogni nodo (comunità) viene calcolato l'attributo 'mean_opinion'
        in base all'opinione media dei nodi della comunità corrispondente.

        Parametri:
        - matches_array: array numpy di matching (ogni riga: (start, end, peso))
        
        :return: grafo evolutivo (NetworkX DiGraph)
        """
        G = nx.DiGraph()
        # Aggiungi gli archi in base ai matching
        for row in matches_array:
            nodo_start, nodo_end, peso = row
            G.add_edge(nodo_start, nodo_end, weight=float(peso))

        # Per ogni comunità nel grafo, calcola e assegna l'attributo "mean_opinion"
        for community in list(G.nodes):
            snapshot_number = int(community.split('_')[0]) 
            clustering = self.tc.get_clustering_at(snapshot_number)
            # Estrae i nodi appartenenti alla community, se non ci sono ritorna una lista vuota
            community_nodes = clustering.named_communities.get(community, [])
            if not community_nodes: # se la lista è vuota l'opinione è 0.5
                mean_opinion = 0.5
            else:
                opinions = [self.snapshot_graphs[snapshot_number].nodes[n]["opinion"]
                            for n in community_nodes]
                mean_opinion = sum(opinions) / len(opinions)
            G.nodes[community]["mean_opinion"] = mean_opinion

        self.evolution_graph = G
        return G

    def plot_evolution_graph(self, use_filtered=True, save=False):
        """
        Visualizza il grafo evolutivo. Se use_filtered è True, usa i matching filtrati;
        altrimenti usa tutti i matching.
        """
        if use_filtered:
            if self.filtered_matches is None:
                self.filter_matches_by_threshold()
            matches_array = self.filtered_matches
        else:
            if self.matches_array is None:
                raise ValueError("Eseguire prima compute_matches().")
            matches_array = self.matches_array

        G = self.community_evolution_graph(matches_array)
        pos = nx.spring_layout(G, seed=42)  # Layout deterministico

        # Estrai i valori di mean_opinion per colorare i nodi
        mean_opinions = np.array([G.nodes[n]["mean_opinion"] for n in G.nodes()])
        node_colors = plt.cm.coolwarm(mean_opinions)

        plt.figure(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, node_size=500,
                node_color=node_colors, cmap=plt.cm.coolwarm,
                font_size=10, edge_color="black")

        # Disegna gli archi con spessore proporzionale al peso
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights],
                               arrowstyle='->', arrowsize=15)

        plt.title("Evoluzione community")
        if save:
            # Salva la figura nella cartella 'plots' se esiste (altrimenti crea la cartella)
            os.makedirs('plots', exist_ok=True)
            plt.savefig(os.path.join('plots', 'Evoluzione_community.png'))
        plt.show()

    def get_category_counts(self):
        """
        Esegue l'analisi sul grafo evolutivo (costruito con tutti i matching) e restituisce,
        per ogni snapshot, il numero di community per categoria in un dizionario.
        
        :return: dict, chiave: snapshot (int), valore: dict con chiavi "pro-Biden", "neutral", "pro-Trump"
        """
        if self.matches_array is None:
            raise ValueError("Eseguire prima compute_matches().")
        G = self.community_evolution_graph(self.matches_array)

        def categorize_opinion(mean_opinion):
            if mean_opinion <= 0.333:
                return "pro-Biden"
            elif mean_opinion >= 0.666:
                return "pro-Trump"
            else:
                return "neutral"

        snapshot_categories = defaultdict(dict)
        for community in G.nodes():
            snapshot_number, _ = community.split('_')
            snapshot_number = int(snapshot_number)
            mean_opinion = G.nodes[community]["mean_opinion"]
            category = categorize_opinion(mean_opinion)
            snapshot_categories[snapshot_number][community] = category

        category_counts_per_snapshot = defaultdict(lambda: {"pro-Biden": 0, "neutral": 0, "pro-Trump": 0})
        for snapshot, communities in snapshot_categories.items():
            for _, category in communities.items():
                category_counts_per_snapshot[snapshot][category] += 1

        return category_counts_per_snapshot

    def analyze_community_evolution_multiple(self, iterations=10, plot=True, save=False):
        """
        Esegue l'analisi dell'evoluzione delle comunità per un numero specificato di iterazioni,
        in modo da ottenere una stima robusta (media e deviazione standard) dei risultati.
        Viene infine visualizzato un plot in cui, per ogni snapshot (settimana), si mostra:
          - La media del numero di community per categoria.
          - Una fascia che rappresenta ± la deviazione standard.
        
        Le funzioni plot_jaccard_histogram e plot_evolution_graph non vengono iterate.
        
        Separiamo il calcolo di averages e stds dal plot, restituendo tali valori
        per poterli eventualmente utilizzare anche fuori dalla classe.

        Parametri:
        - iterations (int): numero di iterazioni da eseguire.
        - plot (bool): se True, viene mostrato il plot finale.
        
        :return: averages (dict), stds (dict)
        """
        num_snapshots = len(self.snapshot_graphs)
        # Inizializza una struttura per salvare i conteggi per ciascuna categoria e snapshot
        results = {
            "pro-Biden": {s: [] for s in range(num_snapshots)},
            "neutral": {s: [] for s in range(num_snapshots)},
            "pro-Trump": {s: [] for s in range(num_snapshots)}
        }
        
        for i in tqdm(range(iterations)):
            # Reinizializza le strutture per il clustering e i matching
            self.tc = TemporalClustering()
            self.matches = None
            self.matches_array = None
            self.filtered_matches = None
            
            # Calcola le clusterizzazioni e i matching
            self.compute_clusterings()
            self.compute_matches()
            
            # Ottieni i conteggi per questa iterazione
            counts = self.get_category_counts()
            for s in range(num_snapshots):
                if s in counts:
                    for category in ["pro-Biden", "neutral", "pro-Trump"]:
                        results[category][s].append(counts[s].get(category, 0))
                else:
                    for category in ["pro-Biden", "neutral", "pro-Trump"]:
                        results[category][s].append(0)
        
        # Calcola la media e la deviazione standard per ogni snapshot e per ogni categoria
        snapshots = np.array([s + 1 for s in range(num_snapshots)])  # Le settimane partono da 1
        averages = {}
        stds = {}
        for category in results:
            averages[category] = np.array([np.mean(results[category][s]) for s in range(num_snapshots)])
            stds[category] = np.array([np.std(results[category][s], ddof=1) for s in range(num_snapshots)])
        
        # Plot finale: evoluzione media con banda di ± deviazione standard (se richiesto)
        if plot:
            plt.figure(figsize=(10, 5))
            colors = {"pro-Biden": "blue", "neutral": "gray", "pro-Trump": "red"}
            for category in ["pro-Biden", "neutral", "pro-Trump"]:
                plt.plot(snapshots, averages[category], marker="o", linestyle="-", label=category, color=colors[category])
                plt.fill_between(snapshots,
                                 averages[category] - stds[category],
                                 averages[category] + stds[category],
                                 color=colors[category], alpha=0.2)
            
            plt.xlabel("Settimana")
            plt.ylabel("Numero di Community")
            plt.title("Evoluzione media delle opinioni nel tempo")
            plt.xticks(snapshots)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            if save:
                # Salva la figura nella cartella 'plots' se esiste (altrimenti crea la cartella)
                os.makedirs('plots', exist_ok=True)
                plt.savefig(os.path.join('plots', 'Evoluzione_opinioni_medie.png'))
            plt.show()
        
        # Restituisce i valori calcolati per poterli utilizzare anche al di fuori della classe
        return averages, stds, results

