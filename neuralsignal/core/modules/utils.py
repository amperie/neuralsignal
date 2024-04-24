import uuid
import pickle


def generate_uuid():
    return str(uuid.uuid4())


def serialize(obj):
    return pickle.dumps(obj)
