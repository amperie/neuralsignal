import sys
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

cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
config = get_config(cfg_path)
run_automation(config)
logging.info("Automation complete")
