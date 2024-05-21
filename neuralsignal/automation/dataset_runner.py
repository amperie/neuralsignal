import logging
from neuralsignal.core.modules.detector import Detector
from neuralsignal.datasets.dataset_runner import DatasetRunner
from neuralsignal.datasets.dataset_creator import DatasetCreator
from neuralsignal.datasets.s1_trainer import S1Trainer
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
        detector = Detector(detector)
        ds.append(detector)
    cfg['detectors'] = ds
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

    dc = DatasetCreator(cfg)
    ds = dc.create_dataset(cfg['query'])
    return ds


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

    mt = S1Trainer(cfg)
    m = mt.train_model()
    return m
