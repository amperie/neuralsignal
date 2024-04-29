from neuralsignal.datasets.dataset import NSDataset


# TODO: Add this dataset: hotpot_qa
# https://huggingface.co/datasets/hotpot_qa

def add_to_dataset_dictionary(dataset_dict, cfg, dataset_list, include):
    ds_name = cfg["dataset_name"]
    if len(dataset_list) == 0 or \
            (ds_name in dataset_list
                and include) or \
            (ds_name not in dataset_list
                and not include):
        dataset_dict[ds_name] = NSDataset(cfg)


def datasets_dictionary(
        row_limit=0, dataset_list=[], include=True, dataset_variant=None)\
        -> dict:
    """Returns a dict of defined datasets in HF
    Keys are the HF dataset names, values are the dataset objects
    Arguments:
        row_limit {int} -- row limit for each dataset
        dataset_list {list(str)} -- list of dataset names to be returned
    Returns:
        dict: dictionary of datasets

        input_processor functions for each dataset shuld return a dict
        with the following content:
            input: input to the model (user's query)
            context: any context sent in with the input
            output: output of the model that is being evaluated
            ground_truth: if there is one
            metadata: dict of any fields that will pass through
    """
    retVal = {}

    # diagram_hivemapper - /data/nfs-data/diagram/diagram_dataset.json
    def input_processor(row):
        context = row['question_and_context']
        output = row['answer']
        # TODO: split up context into question and context
        question = None
        ground_truth = row['judgement']
        if ground_truth == "yes":
            ground_truth = True
        elif ground_truth == "no":
            ground_truth = False
        else:
            ground_truth = "skip_row"

        return {
                "input": question,
                "context": context,
                "output": output,
                "ground_truth": ground_truth,
                "metadata": {}
                }

    cfg = {
        "dataset_name": "diagram_hivemapper",
        "dataset_type": "local_json_one_per_line",
        "local_path": "/data/nfs-data/diagram/diagram_dataset.json",
        "input_processor": input_processor,
        "hf_token": "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",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # allenai toxicity test
    def input_processor(row):
        return {
                "input": row["prompt"],
                "context": "",
                "output": "",
                "ground_truth": row["ground_truth"],
                "metadata": {}
                }

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "allenai_toxicity",
        "local_path": "//data/nfs-data/allenai_toxicity.json",
        "input_processor": input_processor,
        "hf_token": "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",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    return retVal
