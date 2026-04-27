from neo4j import AsyncGraphDatabase
import logging

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
                                  connfidence = confidence,
                                  risk_weight = risk_weight)
                logger.info("Upserted Edge: %s - [ %s ] -> %s (Risk Weight: %.2f)", subj, predicate, obj, risk_weight)

    async def calculate_sna_metrics(self):
        """
        Calculates basic SNA metrics and saves them as properties on the Inmate nodes. (TO be improved*)
        """
        async with self.driver.session() as session:
            # 1. Degree Centrality (Who is the most active?)
            # Calculates how many edges connect to each Inmate and saves it as 'degree_score'
            degree_query = """
            MATCH (i:Inmate)-[r]-()
            WITH i, count(r) AS degree
            SET i.degree_score = degree
            """
            await session.run(degree_query)
            
            # 2. In-Degree Centrality (Who is being targeted/talked about the most?)
            in_degree_query = """
            MATCH ()-[r]->(i:Inmate)
            WITH i, count(r) AS in_degree
            SET i.in_degree_score = in_degree
            """
            await session.run(in_degree_query)
            
            logger.info("SNA Metrics updated successfully.")            