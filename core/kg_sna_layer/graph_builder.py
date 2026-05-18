from neo4j import AsyncGraphDatabase, GraphDatabase
import logging
import networkx as nx
import networkx.algorithms.community as nx_comm

from core.schemas import Taxonomy
from core.schemas import PredicateOntology

logger = logging.getLogger(__name__)

class AsyncKnowledgeGraphManager:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def ingest_log_data(self, processed_json: dict):
        log_id = processed_json.get("id", "UNKNOWN")
        category = processed_json.get("taxonomy_category", Taxonomy.NEUTRAL.value)
        entities = processed_json.get("entities", [])
        relations = processed_json.get("relations", [])
        confidence = processed_json.get("confidence", 1)

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
                
                query = f"MERGE (n:{label} {{name: $name}})"
                # Await the execution of the query
                await session.run(query, name=name)
                logger.info("Upserted Node: (%s) %s", label, name)

            # 2. Upsert Relationships
            for rel in relations:
                subj = str(rel.get("subject"))
                obj = str(rel.get("object"))
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
            MATCH ()-[r]->(i:Inmate)
            WHERE type(r) <> 'SNA_TIE_STRENGTH'
            WITH i, sum(r.risk_weight) AS incoming_danger
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
            G = nx.Graph()
            for record in records:
                G.add_edge(record["source"], record["target"], weight=record["weight"])
                
            # 3. Calculate Advanced Structural Metrics mathematically
            # Betweenness: Who acts as the bridge/smuggler between different gangs?
            betweenness = nx.betweenness_centrality(G, weight='weight')
            
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

def wipe_sna_metrics(uri, user, pwd):
    """
    Clean SNA metrics calculated.
    """
    driver = GraphDatabase.driver(uri, auth=(user , pwd))
    with driver.session() as session:
        # 1. Delete the calculated meta-edges
        session.run("MATCH ()-[r:SNA_TIE_STRENGTH]->() DELETE r")
        print("✅ Deleted SNA_TIE_STRENGTH edges.")
        
        # 2. Remove the mathematical properties from the nodes
        session.run("""
            MATCH (n:Inmate) 
            REMOVE n.betweenness, n.eigenvector, n.reputation_score, n.community_id
        """)
        print("✅ Removed structural metrics (betweenness, eigenvector, etc.) from nodes.")
        

    print("🧹 Graph successfully reset to its pure semantic state!")            