from neo4j import AsyncGraphDatabase
import logging
import networkx as nx

from core.schemas import Taxonomy

logger = logging.getLogger(__name__)

class AsyncKnowledgeGraphManager:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def ingest_log_data(self, processed_json: dict):
        log_id = processed_json.get("id", "UNKNOWN") # Using your Pydantic generated ID
        category = processed_json.get("taxonomy_category", "UNKNOWN")
        entities = processed_json.get("entities", [])
        relations = processed_json.get("relations", [])
        confidence = processed_json.get("confidence", 1)

        base_severity = Taxonomy[category].value
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
                predicate = str(rel.get("predicate", "RELATES_TO")).upper().replace(" ", "_")

                query = f"""
                MATCH (source {{name: $subj}})
                MATCH (target {{name: $obj}})
                MERGE (source)-[r:{predicate}]->(target)
                SET r.log_id = $log_id,
                    r.taxonomy = $category,
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
        Calculates basic SNA metrics and saves them as properties on the Inmate nodes. (TO be improved*)
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
            G = nx.DiGraph()
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

            # Closeness: Who can spread information/contraband the fastest?
            closeness = nx.closeness_centrality(G, distance='weight')

            # 4. Push the Structural Metrics back into Neo4j
            for node_name in G.nodes():
                update_query = """
                MATCH (i:Inmate {name: $name})
                SET i.betweenness = $betw,
                    i.eigenvector = $eigen,
                    i.closeness = $close
                """
                await session.run(update_query, 
                                  name=node_name, 
                                  betw=betweenness.get(node_name, 0.0),
                                  eigen=eigenvector.get(node_name, 0.0),
                                  close=closeness.get(node_name, 0.0))
                
            logger.info("Structural Metrics (Betweenness, Eigenvector, Closeness) updated.")

        
            
            logger.info("SNA Metrics updated successfully.")            