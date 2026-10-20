from neo4j import AsyncGraphDatabase, GraphDatabase
import logging
import math
import networkx as nx
import torch
import networkx.algorithms.community as nx_comm

from core.schemas import EntityOntology, Taxonomy
from core.schemas import PredicateOntology

logger = logging.getLogger(__name__)

class AsyncKnowledgeGraphManager:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def ingest_log_data(self, processed_json: dict):
        log_id = processed_json.get("id", EntityOntology.UNKNOWN.value)
        short_id = str(log_id)[:8]
        category = processed_json.get("taxonomy_category", Taxonomy.NEUTRAL.value)
        entities = processed_json.get("entities", [])
        relations = processed_json.get("relations", [])
        confidence = normalize_confidence(processed_json.get("confidence", 1))

        try:
            base_severity = Taxonomy(category).severity
        except ValueError:
            logger.warning("Invalid taxonomy category '%s'. Defaulting to NEUTRAL.", category)
            category = "NEUTRAL"
            base_severity = Taxonomy.NEUTRAL.severity

        risk_weight = base_severity * confidence

        # Open an async session
        async with self.driver.session() as session:
            # 1. Upsert Entities
            for ent in entities:
                label = str(ent.get("label", "Entity")).replace(" ", "")
                name = str(ent.get("entity", "Unknown"))

                if "DROP" in name.upper() or "GENERIC" in name.upper():
                    continue

                if "Unknown" in name:
                    name = name.replace(" ", "_")
                    name = f"{name}_{short_id}"
                
                query = f"MERGE (n:{label} {{name: $name}})"
                
                # Await the execution of the query
                await session.run(query, name=name)
                logger.info("Upserted Node: (%s) %s", label, name)

            # 2. Upsert Relationships
            for rel in relations:
                subj = str(rel.get("subject"))
                obj = str(rel.get("object"))

                # 🛑 ADDED MISSING FIREWALL: Stop the loop if it's garbage!
                if "DROP" in subj.upper() or "GENERIC" in subj.upper():
                    continue
                if "DROP" in obj.upper() or "GENERIC" in obj.upper():
                    continue
                
                # The Unknown logic matches perfectly!
                if "Unknown" in subj:
                    subj = subj.replace(" ", "_")
                    subj = f"{subj}_{short_id}"
                if "Unknown" in obj:
                    obj = obj.replace(" ", "_")
                    obj = f"{obj}_{short_id}"
                    
                predicate = str(rel.get("predicate", PredicateOntology.ASSOCIATED_WITH.value)).upper().replace(" ", "_").replace("-", "_")

                query = f"""
                MATCH (source {{name: $subj}})
                MATCH (target {{name: $obj}})
                MERGE (source)-[r:`{predicate}` {{log_id: $log_id}}]->(target)
                SET r.taxonomy = $category,
                    r.confidence_score = $confidence,
                    r.risk_weight = $risk_weight 
                """
                # Await the execution of the query
                await session.run(query, 
                                  subj=subj, 
                                  obj=obj, 
                                  log_id=log_id, 
                                  category=category,
                                  confidence = confidence,
                                  risk_weight = risk_weight)
                logger.info("Upserted Edge: %s - [ %s ] -> %s (Risk Weight: %.2f)", subj, predicate, obj, risk_weight)

    async def calculate_sna_metrics(self):
        """
        Calculates basic SNA metrics and saves them as properties on the Inmate nodes.
        """
        async with self.driver.session() as session:

            # =====================
            # RELATIONAL METRICS
            # =====================

            # A. Edge Strength / Tie Strength
            # How intense is the relationship between two specific inmates?
            # We calculate this by summing the risk_weights of all their interactions.
            strength_query = """
            MATCH (a:Inmate)-[r]->(b:Inmate)
            WHERE type(r) <> 'SNA_TIE_STRENGTH'
            WITH a, b, sum(r.risk_weight) AS total_strength, count(r) AS interaction_count
            // We create a hidden 'meta-edge' just to store the relationship strength
            MERGE (a)-[meta:SNA_TIE_STRENGTH]->(b)
            SET meta.weight = total_strength,
                meta.frequency = interaction_count
            """
            await session.run(strength_query)

            # B. Reputation / Target Score
            # If an inmate has many high-risk arrows pointing AT them, their reputation score rises.
            reputation_query = """
            MATCH (i:Inmate)
            OPTIONAL MATCH ()-[r]->(i) WHERE type(r) <> 'SNA_TIE_STRENGTH'
            WITH i, COALESCE(sum(r.risk_weight), 0) AS incoming_danger
            SET i.reputation_score = incoming_danger
            """
            await session.run(reputation_query)

            logger.info("Relational Metrics (Strength & Reputation) updated.")

            # =====================
            # STRUCTURAL METRICS
            # =====================

            # 1. Fetch the graph structure from Neo4j
            fetch_query = """
            MATCH (source:Inmate)-[meta:SNA_TIE_STRENGTH]->(target:Inmate)
            RETURN source.name AS source, target.name AS target, meta.weight AS weight
            """
            result = await session.run(fetch_query)
            records = await result.data()
            
            # If the graph is empty, stop here
            if not records:
                return

            # 2. Build the NetworkX Directed Graph
            G = nx.DiGraph()
            for record in records:
                G.add_edge(record["source"], record["target"], weight=record["weight"])
            
            for u, v, data in G.edges(data=True):
                # Prevent division by zero
                G[u][v]['distance'] = 1.0 / data['weight'] if data['weight'] > 0 else 1.0

            # 3. Calculate Advanced Structural Metrics mathematically
            # Betweenness: Who acts as the bridge/smuggler between different gangs?
            betweenness = nx.betweenness_centrality(G, weight='distance')
            
            # Eigenvector: Who is connected to OTHER highly connected/dangerous people? (The Boss metric)
            try:
                eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {node: 0.0 for node in G.nodes()} # Fallback if graph is too small/disconnected

            communities = nx_comm.louvain_communities(G, weight='weight')

            # We need to map these sets into a dictionary so we can push it to Neo4j
            # e.g., {"Viper": "Comm_0", "Runner": "Comm_0", "Miller": "Comm_1"}
            community_map = {}
            for comm_id, group in enumerate(communities):
                for inmate in group:
                    community_map[inmate] = f"Subgroup_{comm_id}"

            # Closeness: Who can spread information/contraband the fastest?
            #closeness = nx.closeness_centrality(G, distance='weight')

            # 4. Push the Structural Metrics back into Neo4j
            for node_name in G.nodes():
                update_query = """
                MATCH (i:Inmate {name: $name})
                SET i.betweenness = $betw,
                    i.eigenvector = $eigen,
                    i.community_id = $comm_id
                """
                await session.run(update_query, 
                                name=node_name, 
                                betw=betweenness.get(node_name, 0.0),
                                eigen=eigenvector.get(node_name, 0.0),
                                comm_id=community_map.get(node_name, "Unknown"))                
            logger.info("Structural Metrics (Betweenness, Eigenvector, Closeness) updated.")      
            print("✅ Structural Metrics (Betweenness, Eigenvector, Closeness) updated.")
            logger.info("SNA Metrics updated successfully.")

    async def calculate_topological_edge_metrics(self):
        """
        Calculates shared-space topological metrics (Jaccard, Common Neighbors, Adamic-Adar)
        and stores them directly on the SNA_TIE_STRENGTH edges in Neo4j.
        """
        logger.info("Starting Topological Edge Feature calculation...")
        
        async with self.driver.session() as session:
            # 1. Fetch the existing edges
            fetch_query = """
            MATCH (source:Inmate)-[meta:SNA_TIE_STRENGTH]->(target:Inmate)
            RETURN source.name AS source, target.name AS target
            """
            result = await session.run(fetch_query)
            records = await result.data()
            
            if not records:
                logger.warning("No SNA_TIE_STRENGTH edges found. Run calculate_sna_metrics first.")
                return

            # 2. Build an Undirected NetworkX Graph 
            # (Math requirement: Jaccard & Adamic-Adar require undirected topology)
            G_nx = nx.Graph()
            for record in records:
                G_nx.add_edge(record["source"], record["target"])

            # Helper for safe Adamic-Adar (prevents division by zero)
            def safe_adamic_adar(G, u, v):
                aa = 0.0
                for w in nx.common_neighbors(G, u, v):
                    deg = G.degree(w)
                    if deg > 1:
                        aa += 1 / math.log(deg)
                return aa

            # Pre-calculate Jaccard to speed up the loop
            edges_list = list(G_nx.edges())
            jaccard_generator = nx.jaccard_coefficient(G_nx, edges_list)
            j_dict = {(u, v): p for u, v, p in jaccard_generator}

            # 3. Calculate and push metrics back to Neo4j edge-by-edge
            update_count = 0
            for record in records:
                u = record["source"]
                v = record["target"]
                
                # Calculate the 3 metrics
                cn = len(list(nx.common_neighbors(G_nx, u, v)))
                jac = j_dict.get((u, v), j_dict.get((v, u), 0.0))
                aa = safe_adamic_adar(G_nx, u, v)

                # Push directly onto the existing meta-edge in Neo4j
                update_query = """
                MATCH (source:Inmate {name: $source})-[meta:SNA_TIE_STRENGTH]->(target:Inmate {name: $target})
                SET meta.common_neighbors = $cn,
                    meta.jaccard = $jac,
                    meta.adamic_adar = $aa
                """
                await session.run(update_query, source=u, target=v, cn=cn, jac=jac, aa=aa)
                update_count += 1

            logger.info("Topological Edge Features successfully saved to %d edges.", update_count)
            print("✅ Topological Edge Features (Jaccard, CN, Adamic-Adar) updated in Neo4j.")

def normalize_confidence(score: float) -> float:
    score = float(score)
    if score > 1.0 and score <= 100:
        return score / 100.0
    return max(0.0, min(1.0, score))

def wipe_sna_metrics(uri, user, pwd):
    """
    Clean SNA metrics calculated.
    """
    driver = GraphDatabase.driver(uri, auth=(user , pwd))
    with driver.session() as session:
        # 1. Delete the calculated meta-edges (This wipes Tie Strength AND Topological Edge Features)
        session.run("MATCH ()-[r:SNA_TIE_STRENGTH]->() DELETE r")
        print("✅ Deleted SNA_TIE_STRENGTH edges (including Jaccard, CN, and Adamic-Adar).")
        
        # 2. Remove the mathematical properties from the nodes
        session.run("""
            MATCH (n:Inmate) 
            REMOVE n.betweenness, n.eigenvector, n.reputation_score, n.community_id
        """)
        print("✅ Removed structural metrics (betweenness, eigenvector, etc.) from nodes.")
        
    print("🧹 Graph successfully reset to its pure semantic state!") 

def calculate_strict_fold_topology_original(reference_edge_index, target_edge_index, num_nodes, edge_weights=None):
    # 1. Build the Safe Training Graph (Directed)
    G_ref = nx.DiGraph() 
    G_ref.add_nodes_from(range(num_nodes))
    
    edges = reference_edge_index.t().tolist()
    
    if edge_weights is not None:
        weights = edge_weights.tolist()
        weighted_edges = [(u, v, {'weight': w, 'distance': 1.0 / w if w > 0 else 1.0}) for (u, v), w in zip(edges, weights)]
        G_ref.add_edges_from(weighted_edges)
        
        # NLP Context: Source = Aggressiveness (Outgoing), Target = Reputation (Incoming)
        reputation = {n: 0.0 for n in range(num_nodes)}
        aggressiveness = {n: 0.0 for n in range(num_nodes)}
        for u, v, data in G_ref.edges(data=True):
            w = data.get('weight', 0.0)
            aggressiveness[u] += w 
            reputation[v] += w    
    else:
        G_ref.add_edges_from(edges)

    # 2. NODE METRICS 
    betweenness = nx.betweenness_centrality(G_ref, weight='distance' if edge_weights is not None else None)
    try:
        eigenvector = nx.eigenvector_centrality(G_ref, max_iter=1000, weight='weight' if edge_weights is not None else None)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G_ref.nodes()}

    # 3. EDGE METRICS 
    G_un = G_ref.to_undirected() # Required for Common Neighbors & Jaccard
    target_edges = target_edge_index.t().tolist()
    edge_features = []
    
    for u, v in target_edges:
        # Relational Edge Topology (Undirected)
        cn = len(list(nx.common_neighbors(G_un, u, v)))
        try:
            preds = nx.jaccard_coefficient(G_un, [(u, v)])
            jac = next(preds)[2]
        except ZeroDivisionError:
            jac = 0.0
            
        base_feats = [
            cn, 
            jac, 
            betweenness.get(u, 0.0), 
            betweenness.get(v, 0.0),
            eigenvector.get(u, 0.0),
            eigenvector.get(v, 0.0)
        ]
        
        # If Enriched, add the 2 NLP Edge Features
        if edge_weights is not None:
            base_feats.extend([aggressiveness.get(u, 0.0), reputation.get(v, 0.0)])
            
        edge_features.append(base_feats)
        
    return torch.tensor(edge_features, dtype=torch.float)       

#Corrected
def calculate_strict_fold_topology(reference_edge_index, target_edge_index, num_nodes, edge_weights=None):
    G_ref = nx.DiGraph()
    G_ref.add_nodes_from(range(num_nodes))
    edges = reference_edge_index.t().tolist()

    if edge_weights is not None:
        weights = edge_weights.tolist()
        aggressiveness = [0.0] * num_nodes      # sum of outgoing weights (raw edge list)
        reputation = [0.0] * num_nodes          # sum of incoming weights (raw edge list)
        tie = {}                                # parallel edges accumulate into one tie
        for (u, v), w in zip(edges, weights):
            aggressiveness[u] += w
            reputation[v] += w
            tie[(u, v)] = tie.get((u, v), 0.0) + w
        G_ref.add_edges_from(
            (u, v, {'weight': w, 'distance': 1.0 / w if w > 0 else 1e6}) for (u, v), w in tie.items())
    else:
        G_ref.add_edges_from(edges)

    use_w = edge_weights is not None
    betweenness = nx.betweenness_centrality(G_ref, weight='distance' if use_w else None)
    try:
        eigenvector = nx.eigenvector_centrality(G_ref, max_iter=1000, weight='weight' if use_w else None)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G_ref.nodes()}

    G_un = G_ref.to_undirected()
    edge_features = []
    for u, v in target_edge_index.t().tolist():
        cn = len(list(nx.common_neighbors(G_un, u, v)))
        try:
            jac = next(nx.jaccard_coefficient(G_un, [(u, v)]))[2]
        except ZeroDivisionError:
            jac = 0.0
        feats = [cn, jac, betweenness.get(u, 0.0), betweenness.get(v, 0.0),
                 eigenvector.get(u, 0.0), eigenvector.get(v, 0.0)]
        if use_w:
            feats.extend([aggressiveness[u], reputation[v]])
        edge_features.append(feats)
    return torch.tensor(edge_features, dtype=torch.float)