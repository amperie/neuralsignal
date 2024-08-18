import logging
import yaml
import platform
import sys
from neuralsignal.core.modules.detector import Detector
from neuralsignal.datasets.dataset_runner import DatasetRunner
from neuralsignal.datasets.dataset_creator import DatasetCreator
from neuralsignal.datasets.s1_trainer import S1Trainer
from neuralsignal.core.modules.feature_sets.feature_processor\
    import FeatureProcessor
from neuralsignal.core.modules.utils import get_name_from_template
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


def run_data_collection(cfg: dict):
    """Opens a defined dataset and runs data collection for it

    Args:
        cfg (dict): Configuration should include:
        application_name (str): Name of the application
        sub_application_name (str): Name of the sub application
        dataset_name (str): Name of the dataset
        detector_names (list): List of detector names
    Optional:
        max_new_tokens (int): Maximum number of new tokens
        row_limit (int): Maximum number of rows to run
        batch_size (int): Batch size
    """
    # Get all the detectors from config
    ds = []
    for d in cfg['detector_names']:
        detector = sdk_config.get_detector_config(d)
        if detector['enabled']:
            detector['application_name'] = cfg['application_name']
            detector['sub_application_name'] = cfg['sub_application_name']
            detector = Detector(detector)
            ds.append(detector)
    cfg['detectors'] = ds
    cfg['row_limit'] = cfg['data_collection_row_limit']
    dsr = DatasetRunner(cfg)
    dsr.run()


def create_dataset(cfg: dict):
    """Creates a dataset from a query of scans

    Args:
        cfg (dict): Config should include:
        application_name (str): Name of the application
        sub_application_name (str): Name of the sub application
        write_to_file (bool): Write to file
        build_in_memory (bool): Build in memory
        file_out (str): Name of the output file
        detector_name (str): Name of the detector
        query (dict): Query of the scans to create dataset for
    Optional:
        zone_size (int): Zone size
        row_limit (int): Maximum number of rows to run
        write_header (bool): Write header
        use_full_zone_names (bool): Use full zone names
        include_output (bool): Include output
        use_gt_as_target (bool): Use gt as target
        tensor_field_to_use (str): Tensor field to use
    Returns:
        Returns a tuple:
        retVal[0] = path to file out if write_to_file is True
        retVal[1] = pandas dataframe if build_in_memory is True
    """

    cfg['row_limit'] = cfg['dataset_row_limit']
    dataset_paths = []
    file_out_template = cfg['file_out']

    for d in cfg['detector_names']:
        detector = sdk_config.get_detector_config(d)
        if detector['enabled']:
            cfg['detector_name'] = d
            # file_out = file_out_template.replace("{detector}", d)

            file_out = get_name_from_template(
                file_out_template,
                {
                    "dataset": cfg['dataset'],
                    "detector": d,
                    "model": cfg['indirect_config']["indirect_model"]
                }
                )

            cfg['file_out'] = file_out
            fsc = cfg['feature_set_configs']
            fp = FeatureProcessor(feature_set_configs=fsc)
            cfg['feature_processor'] = fp

            dc = DatasetCreator(cfg)
            query = cfg['query']
            dc.create_dataset(query)
            dataset_paths.append(file_out)
    # Set this back to the original template for later stages
    cfg['file_out'] = file_out_template
    return dataset_paths


def create_s1_model(cfg: dict):
    """Creates a S1 model from a dataset

    Args:
        cfg (dict): Config should include:
        application_name (str): Name of the application
        sub_application_name (str): Name of the sub application
        model_name (str): Name of the model
        (one of these two should be present)
        dataset_path (str): Path to the dataset
        dataset_dataframe (Dataframe): Dataframe of the dataset
        save_to_backend (bool): Save to backend
    Optional:
        description (str): Description of the model
        params(dict): Parameters of the model
        tags (dict): Tags of the model
        metadata (dict): Metadata of the model
    """

    # Iterate through all the detectors
    models = []
    file_out_template = cfg['file_out']
    for d in cfg['detector_names']:
        detector = sdk_config.get_detector_config(d)
        if detector['enabled']:
            for rl in cfg['modeling_row_limits']:
                # Set up modeling parameters
                cfg['row_limit'] = rl
                cfg['detector_name'] = d

                file_out = get_name_from_template(
                    file_out_template,
                    {
                        "dataset": cfg['dataset'],
                        "detector": d,
                        "model": cfg['indirect_config']["indirect_model"]
                    }
                    )
                cfg['dataset_path'] = file_out

                cfg['model_name'] = cfg['model_name'] + "_" + d
                d_cfg = sdk_config.get_detector_config(d)
                cfg['params'] = {
                    "zone_size": cfg['zone_size'],
                    "row_limit": cfg['row_limit'],
                    "use_full_zone_names": cfg['use_full_zone_names'],
                    "model": cfg['indirect_config']['indirect_model'],
                    "quantization": cfg['indirect_config']['quantization'],
                    "dataset": cfg['dataset'],
                    "detector": d,
                    "s1_model_for_detector": d_cfg['S1_model'],
                    "prompt": d_cfg['prompt'],
                }
                cfg['run_name'] = cfg['model_name']
                mt = S1Trainer(cfg)
                m = mt.train_model()
                models.append(m)
    return models


def run_automation(cfg: dict):
    """Runs the automation

    Args:
        cfg (dict): Config should include:
        application_name (str): Name of the application
        sub_application_name (str): Name of the sub application
        run_data_collection (bool): Run data collection
        create_dataset (bool): Create dataset
        create_s1_model (bool): Create S1 model
    """
    if cfg['run_data_collection']:
        run_data_collection(cfg)
    if cfg['create_dataset']:
        create_dataset(cfg)
        cfg['dataset_path'] = cfg['file_out']
    if cfg['create_s1_model']:
        cfg['row_limit'] = cfg['modeling_row_limit']
        create_s1_model(cfg)


##########################################################
# CONFIG AND SETUP                                       #
##########################################################

try:
    if platform.system() == "Windows" or platform.system() == "Darwin":
        yaml_config =\
            "neuralsignal/neuralsignal/automation/dataset_automation.yaml"
    else:
        yaml_config = sys.argv[1]
except IndexError:
    raise ValueError(
        "Usage: python dataset_automation.py <yaml config file>")

# Get config from YAML
config = yaml.safe_load(open(yaml_config))
run_automation(config)
logging.info("Automation complete")
