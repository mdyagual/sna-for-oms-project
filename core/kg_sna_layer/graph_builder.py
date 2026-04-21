from neo4j import AsyncGraphDatabase
import logging

class AsyncKnowledgeGraphManager:
    def __init__(self, uri, user, password):
        # Initialize the ASYNC driver
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def ingest_log_data(self, processed_json: dict):
        log_id = processed_json.get("id", "UNKNOWN") # Using your Pydantic generated ID
        category = processed_json.get("taxonomy_category", "UNKNOWN")
        entities = processed_json.get("entities", [])
        relations = processed_json.get("relations", [])

        # Open an async session
        async with self.driver.session() as session:
            # 1. Upsert Entities
            for ent in entities:
                label = str(ent.get("label", "Entity")).replace(" ", "")
                name = str(ent.get("entity", "Unknown"))
                
                query = f"MERGE (n:{label} {{name: $name}})"
                # Await the execution of the query
                await session.run(query, name=name)
                logging.info(f"Upserted Node: ({label}) {name}")

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
                    r.taxonomy = $category
                """
                # Await the execution of the query
                await session.run(query, subj=subj, obj=obj, log_id=log_id, category=category)
                logging.info(f"Upserted Edge: {subj} -[{predicate}]-> {obj}")