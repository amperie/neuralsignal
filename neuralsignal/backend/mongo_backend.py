import logging
import pymongo
from bson.objectid import ObjectId
import gridfs
import pickle
import io
import torch
import copy
from neuralsignal.core.modules.utils import serialize
from neuralsignal.core.modules.tensors import copy_scan_to_device
from neuralsignal.backend.backend_util import reduce_hd_cache
from neuralsignal.backend.backend_util import save_scan_to_disk
from neuralsignal.backend.backend_util import load_scan_from_disk
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())
logging.getLogger("pymongo").setLevel(logging.ERROR)


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

    # Define this as a static variable so it can be shared across instances
    scan_cache = {}
    scan_hd_cache = []

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
            if "scan_cache_size" in config:
                self.scan_cache_size = config["scan_cache_size"]
                self.scan_hd_cache_size = config["scan_hd_cache_size"]
                self.scan_cache_directory = config["scan_cache_directory"]
            else:
                self.scan_cache_size = 0
                self.scan_hd_cache_size = 0
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

    def build_query_with_detector(self, query: dict):
        q = copy.deepcopy(query)
        d = q['detector_name']
        q.pop('detector_name')
        q[f"detections.{d}"] = {"$ne": None}
        return q

    def check_cache(self, scan: dict):
        """
        The whole scan object has to be sent in as an argument
        Check if the scan is in the cache. If it is, return the cached version
        Otherwise, return None
        """
        _id = str(scan["_id"])
        if _id in MongoBackend.scan_cache:
            # Check the memory cache first
            logging.debug(f"Mongo memory cache hit: {_id}")
            retVal = MongoBackend.scan_cache[_id]
            if "original_device" in retVal:
                retVal = copy_scan_to_device(retVal, retVal["original_device"])
            return retVal
        elif _id in MongoBackend.scan_hd_cache:
            # Check the hard drive cache
            logging.debug(f"Mongo hard drive cache hit: {_id}")
            retVal = load_scan_from_disk(_id, self.scan_cache_directory)
            if "original_device" in retVal:
                retVal = copy_scan_to_device(retVal, retVal["original_device"])
            return retVal
        else:
            return None

    def _insert_to_memory_cache(self, scan: dict):
        _id = str(scan["_id"])
        scan["original_device"] =\
            next(iter(scan['outputs'].values())).device
        MongoBackend.scan_cache[_id] = scan

    def _insert_to_hd_cache(self, scan: dict):
        _id = str(scan["_id"])
        scan["original_device"] =\
            next(iter(scan['outputs'].values())).device
        save_scan_to_disk(scan, self.scan_cache_directory)
        MongoBackend.scan_hd_cache.append(_id)

    def _route_to_cache(self, scan: dict):
        # Logic that decides what cache to write scan to
        # Starting simple and just filling up memory cache
        # Then filling up hard drive cache. That's it

        cache_usage = len(self.scan_cache.keys())
        hd_cache_usage = len(self.scan_hd_cache)

        if self.scan_cache_size > 0:
            cached_scan = copy_scan_to_device(scan, "cpu")
            # We are using cache
            if cache_usage < self.scan_cache_size:
                # Memory cache
                self._insert_to_memory_cache(cached_scan)
            elif hd_cache_usage < self.scan_hd_cache_size:
                # Hard drive cache
                self._insert_to_hd_cache(cached_scan)
            else:
                # Both caches are full
                # Insert to memory cache
                # Move the oldest scan to the hard drive cache
                # Then reduce the disk cache
                self._insert_to_memory_cache(cached_scan)
                oldest = next(iter(MongoBackend.scan_cache.keys()))
                oldest_scan = MongoBackend.scan_cache.pop(oldest)
                self._insert_to_hd_cache(oldest_scan)
                reduce_hd_cache()

    def add_to_cache(self, scan: dict):
        cache_usage = len(self.scan_cache.keys())
        hd_cache_usage = len(self.scan_hd_cache)

        # Are we using the cache?
        if self.scan_cache_size > 0:
            _id = str(scan["_id"])
            scan["original_device"] =\
                next(iter(scan['outputs'].values())).device
            cached_scan = copy_scan_to_device(scan, "cpu")
            if cache_usage < self.scan_cache_size:
                # Still have room in the cache so insert
                MongoBackend.scan_cache[_id] = cached_scan
            else:
                # Remove the oldest scan in the cache first
                (k := next(iter(self.scan_cache)), self.scan_cache.pop(k))
                MongoBackend.scan_cache[_id] = cached_scan
        if cache_usage % 10 == 0:
            logging.debug(f"Mongo cache usage: {cache_usage}")

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

    def deserialize_scan(self, scan):
        if "inputs" in scan:
            scan["inputs"] = self.read_serialized_from_GridFS(
                scan["inputs"])
        if "outputs" in scan:
            scan["outputs"] = self.read_serialized_from_GridFS(
                scan["outputs"])
        return scan

    def load_scan(self, scan_id: str):
        id = ObjectId(scan_id)
        data = self.col.find_one({"_id": id})
        if data is None:
            return None
        cs = self.check_cache(data)
        if cs is None:
            scan = self.deserialize_scan(data)
            self.add_to_cache(scan)
            return scan
        else:
            return cs

    def query(self, query: dict) -> list:
        return self.col.find(query)

    def get_query_count(self, query: dict) -> int:
        return self.col.count_documents(query)

    def iterate_scans(self, query: dict, row_limit: int = 0):
        q = self.build_query_with_detector(query)
        if row_limit == 0:
            mng_cursor = self.query(q)
        else:
            mng_cursor = self.query(q).limit(row_limit)
        for scan in mng_cursor:
            cs = self.check_cache(scan)
            if cs is None:
                scan = self.deserialize_scan(scan)
                self.add_to_cache(scan)
            else:
                scan = cs
            yield scan

    def get_scan_iterator_count(self, query: dict) -> int:
        q = self.build_query_with_detector(query)
        return self.get_query_count(q)

    def load_s1_model(self, model_id: str):
        raise NotImplementedError
