import logging
import pymongo
from bson.objectid import ObjectId
import gridfs
import pickle
import io
import torch
from neuralsignal.core.modules.generation_instance import GenerationInstance
from neuralsignal.core.modules.utils import serialize

logging.basicConfig(level=logging.INFO)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def loads(x):
    bs = io.BytesIO(x)
    unpickler = CPU_Unpickler(bs)
    return unpickler.load()


class MongoBackend:
    """
    Implementation of a Mongo backend. Don't use this class directly
    Configuration options:
        - url: url of the mongo server
        - db: database name
        - collection: collection name
    """
    def __init__(self, config: dict) -> None:
        try:
            self.config = config
            self.mongo_url = config['mongo_url']
            self.db = config['db']
            self.col = config['col']
            self.client = pymongo.MongoClient(self.mongo_url)
            self.db = self.client[self.db]
            self.col = self.db[self.col]
        except KeyError as e:
            raise ValueError(f"Missing configuration parameter {e}")
        except Exception as e:
            raise ValueError(f"Error connecting to Mongo: {e}")

    def write_dict_to_mongo(self, dict_in) -> ObjectId:
        x = self.col.insert_one(dict_in)
        return x.inserted_id

    def save_scan(self, scan: GenerationInstance) -> ObjectId:
        self.write_dict_to_mongo(scan.get_flattened_data())

    def load_scan(self, scan_id: str) -> GenerationInstance:
        pass

    def query(self, query: dict) -> list:
        pass
