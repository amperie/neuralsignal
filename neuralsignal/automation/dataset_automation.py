import logging
from neuralsignal.automation.dataset_automation_core\
    import run_automation
from neuralsignal.automation.dataset_automation_core\
    import get_config
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())

##########################################################
# CONFIG AND SETUP                                       #
##########################################################

config = get_config()
run_automation(config)
logging.info("Automation complete")
