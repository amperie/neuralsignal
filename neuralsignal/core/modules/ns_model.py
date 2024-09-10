import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class NSModel:

    def __init__(self, config: dict) -> None:
        self.config = config