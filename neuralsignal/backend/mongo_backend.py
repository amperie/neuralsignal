import logging
import pymongo
from bson.objectid import ObjectId
from neuralsignal.core.modules.generation_instance import GenerationInstance

logging.basicConfig(level=logging.INFO)


class MongoBackend:
    """
    Implementation of a Mongo backend. Don't use this class directly
    Configuration options:
        - url: url of the mongo server
        - db: database name
        - collection: collection name
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        self.mongo_url = config['url']
        self.db = config['db']
        self.col = config['collection']
        self.client = pymongo.MongoClient(self.mongo_url)
        self.db = self.client[self.db]
        self.col = self.db[self.col]

    def write_dict_to_mongo(self, dict_in) -> ObjectId:
        x = self.col.insert_one(dict_in)
        return x.inserted_id

    def save_scan(self, scan: GenerationInstance) -> ObjectId:
        self.write_dict_to_mongo(scan.get_flattened_data())

    def load_scan(self, scan_id: str) -> GenerationInstance:
        pass

    def query(self, query: dict) -> list:
        pass