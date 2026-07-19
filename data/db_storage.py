import logging

from tinydb import TinyDB, Query
from core.config import demo


db = TinyDB(demo.DB_PAPER, indent=4)
Record = Query()

logger = logging.getLogger(__name__)
async def check_cache(text):
    cached_result = db.search((Record.log_text == text))    
    if cached_result:
        logger.info("Found in caché. Skipping...")
        return True
    return False

async def save_in_cache(d_text):
    db.insert(d_text)
    logger.info("Processed result stored in caché.")

async def get_all_cache():
    return db.all()