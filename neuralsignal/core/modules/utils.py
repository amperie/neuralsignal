import uuid
import pickle
import re
import unicodedata
import os


def generate_uuid():
    return str(uuid.uuid4())


def serialize(obj):
    return pickle.dumps(obj)


def string_to_filename(value: str) -> str:
    file_name = os.path.splitext(value)[0]
    file_ext = os.path.splitext(value)[1]

    value = str(file_name)
    value = unicodedata.normalize('NFKD', value).\
        encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '_', value.lower())
    value = re.sub(r'[-\s]+', '-', value).strip('-_')
    return value + file_ext


def get_name_from_template(
        template: str, replacements: dict, file_safe: bool = True
        ):
    retVal = template
    for s in replacements.keys():
        r = replacements[s]
        if isinstance(r, list):
            retVal = retVal.replace("{" + s + "}", '-'.join(r))
        else:
            retVal = retVal.replace("{" + s + "}", r)
    if file_safe:
        retVal = string_to_filename(retVal)
    return retVal
