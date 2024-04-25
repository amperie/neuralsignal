import uuid
import pickle
import re
import unicodedata


def generate_uuid():
    return str(uuid.uuid4())


def serialize(obj):
    return pickle.dumps(obj)


def string_to_filename(value: str) -> str:
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '_', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
