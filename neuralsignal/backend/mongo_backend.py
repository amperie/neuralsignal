import logging
import pymongo
from bson.objectid import ObjectId
import gridfs
import pickle
import io
import torch
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
        self.config = config
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

    def write_file_to_GridFS(self, file_path):
        fs = gridfs.GridFS(self.db)
        with open(file_path, "rb") as f:
            id = fs.put(f, filename=file_path)
            return id

    def read_file_from_GridFS(self, id):
        fs = gridfs.GridFS(self.db)
        return fs.get(id).read()

    def write_serialized_to_GridFS(self, obj):
        fs = gridfs.GridFS(self.db)
        id = fs.put(obj)
        return id

    def read_serialized_from_GridFS(self, id):
        fs = gridfs.GridFS(self.db)
        f = fs.get(id).read()
        try:
            return pickle.loads(f)
        except RuntimeError:
            return loads(f)

    # Interface methods

    def save_scan(self, scan) -> ObjectId:
        data = scan.get_flattened_data()
        if "outputs" in data:
            data["outputs"] =\
                self.write_serialized_to_GridFS(serialize(data["outputs"]))
        if "inputs" in data:
            data["inputs"] =\
                self.write_serialized_to_GridFS(serialize(data["inputs"]))
        return self.write_dict_to_mongo(data)

    def load_scan(self, scan_id: str):
        raise NotImplementedError

    def query(self, query: dict) -> list:
        raise NotImplementedError

    def load_s1_model(self, model_id: str):
        raise NotImplementedError
