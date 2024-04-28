import random
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
    """
    retVal = {}

    def change_gt(val):
        if val == -1:
            return "negative"
        if val == 0:
            return "neutral"
        return "positive"

    cfg = {
        "column_map": {
            "input_column": "sentence",
            "gt_column": "polarity"
        },
        "gt_processor": change_gt,
        "dataset_name": "fhamborg/news_sentiment_newsmtsc",
        "hf_token": "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",
        "prompt":
            "tell me if the sentiment of this text is negative,\
                neutral or positive: ",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    def change_gt(val):
        if val == 0:
            return "negative"
        if val == 1:
            return "neutral"
        return "positive"

    cfg = {
        "column_map": {
            "input_column": "sentence",
            "gt_column": "label"
        },
        "gt_processor": change_gt,
        "dataset_name": "financial_phrasebank",
        "hf_token": "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",
        "prompt":
            "tell me if the sentiment of this text is negative, \
                neutral or positive: ",
        "config_name":
            "sentences_50agree",
        "split": "train",
        "row_limit": row_limit,
    }
    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    def change_gt(val):
        if val == 0:
            return "Human Necessities"
        if val == 1:
            return "Performing Operations; Transporting"
        if val == 2:
            return "Chemistry; Metallurgy"
        if val == 3:
            return "Textiles; Paper"
        if val == 4:
            return "Fixed Constructions"
        if val == 5:
            return "Mechanical Engineering"
        if val == 6:
            return "Physics"
        if val == 7:
            return "Electricity"
        if val == 8:
            return "General tagging of new or cross-sectional technology"
        return "positive"

    cfg = {
        "column_map": {
            "input_column": "text",
            "gt_column": "label"
        },
        "gt_processor": change_gt,
        "dataset_name": "ccdv/patent-classification",
        "hf_token": "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",
        "prompt":
            "tell me what topic this sentence belongs to - physics, "
            "electricity, human necessities, Performing Operations, "
            "Transporting, General new technology, Chemistry, Metallurgy, "
            "Mechanical Engineering, Lightning, Heating, Weapons, Blasting, "
            "Fixed Constructions, Textiles, Paper: ",
        "config_name":
            None,
        "split": "train+validation+test",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    def input_processor(row):
        question = row["question"]
        context = row["context"]["contexts"][0]
        return "Answer the following question with yes, no or maybe with the "\
            f"context given. Question: {question} Context: {context}"

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "final_decision"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "pubmed_qa",
        "hf_token": "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",
        "prompt": "",
        "config_name": "pqa_artificial",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    def input_processor(row):
        question = row["question"]
        dialogue = row["dialogue"]
        choice = row["choice"]
        return (
            f"Answer the question given the dialogue between people "
            f"and the possible answers. Tell me from the list of possible "
            f"answers which one you think is correct.\nQuestion: {question} \n"
            f"Dialogue:\n{dialogue}\n"
            f"Possible answers: \n{choice}"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answer"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "dream",
        "hf_token": "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",
        "prompt": "",
        "split": "train+test+validation",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    def input_processor(row):
        question = row["question"]
        context = row["context"]
        return (
            f"Please answer the question with the context given by choosing "
            f"from the four possible answers. Just answer with the number of "
            f"the answer you choose. Context: {context} \n"
            f"Question: {question}\n"
            f"Answer #0: {row['answer0']} \n"
            f"Answer #1: {row['answer1']} \n"
            f"Answer #2: {row['answer2']} \n"
            f"Answer #3: {row['answer3']}"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "label"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "cosmos_qa",
        "hf_token": "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",
        "prompt": "",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # google_boolq
    def input_processor(row):
        question = row["question"]
        context = row["passage"]
        return (
            f"Please answer the question with true or false "
            f"given the context \n"
            f"Context: {context} \n"
            f"Question: {question}\n"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answer"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "google/boolq",
        "hf_token": "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",
        "prompt": "",
        "split": "train+validation",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # awua_rat
    def input_processor(row):
        question = row["question"]
        answers = row["options"]
        return (
            f"Answer the question by choosing the the available answers. "
            f"Please answer by only giving the letter of the answer "
            f"and nothing else. \n"
            f"Question: {question}\n"
            f"Possible answers: {answers} \n"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "correct"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "aqua_rat",
        "hf_token": "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",
        "prompt": "",
        "config_name": "raw",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # drop dataset
    def change_gt(val):
        spans = val['spans']
        return spans

    def input_processor(row):
        passage = row["passage"]
        question = row["question"]
        return (
            "Answer the question by reading the passage and choosing the "
            "correct answer or calculating it based on information in the "
            "passage. If there are more than answer, answer with a comma "
            "separated list of answers."
            "Please rspond with the answers and nothing else \n"
            f"Passage: {passage}\n"
            f"Question: {question} \n"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answers_spans"
        },
        "gt_processor": change_gt,
        "input_processor": input_processor,
        "dataset_name": "drop",
        "hf_token": "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",
        "prompt": "",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # super_glue
    def input_processor(row):
        context = row["passage"]
        passage = row["query"]
        return (
            f'Based on this context, replace "@placeholder" with '
            f'the name of the entity or entities '
            f'that make sense in the text.Answer only with the '
            f'word or words you use to replace @placeholder\n'
            f'context: {context}\n'
            f'passage: {passage}'
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answers"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "super_glue",
        "hf_token": "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",
        "prompt": "",
        "config_name": "record",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # cais/mmlu
    def input_processor(row):
        question = row["question"]
        choices = row["choices"]

        return (
            f'From the question, choose the right answer '
            f'Answer only with the letter of the answer and nothing else\n'
            f'question: {question}\n'
            f"Answer A: {choices[0]} \n"
            f"Answer B: {choices[1]} \n"
            f"Answer C: {choices[2]} \n"
            f"Answer D: {choices[3]}"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answer"
        },
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "cais/mmlu",
        "hf_token": "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",
        "prompt": "",
        "config_name": "all",
        "split": "test",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # narrativeqa
    def input_processor(row):
        question = row["question"]["text"]
        text = row["document"]["text"]

        return (
            f'Answer the question based on the context given. \n'
            f"question: {question}"
            f'context: {text}\n'
        )

    def gt_processor(val):
        retVal = []
        for v in val:
            retVal.append(v['text'])
        return retVal

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answers"
        },
        "gt_processor": gt_processor,
        "input_processor": input_processor,
        "dataset_name": "narrativeqa",
        "hf_token": "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",
        "prompt": "",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # hotpot_qa
    def input_processor(row):
        extra_context = 4
        question = row["question"]
        text = []
        sup_facts = row["supporting_facts"]["title"]
        sup_facts_idxs = []
        for s_f in sup_facts:
            idx = row["context"]["title"].index(s_f)
            sup_facts_idxs.append(idx)
            sentences = row["context"]["sentences"][idx]
            text.append(".".join(sentences))
        for ec in row["context"]["sentences"]:
            if row["context"]["sentences"].index(ec) not in sup_facts_idxs:
                text.append(".".join(ec))
                extra_context -= 1
                if extra_context == 0:
                    break
        random.shuffle(text)
        text = " ".join(text)

        return (
            f'Answer the question based on the context given. \n'
            f"question: {question}\n"
            f'context: {text}'
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answer"
        },
        "input_processor": input_processor,
        "dataset_name": "hotpot_qa",
        "hf_token": "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",
        "prompt": "",
        "config_name": "distractor",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # hotpot_qa_hallucinated
    def input_processor(row):
        context = row["input"]\
            .replace("Answer the question based on the context given.", "")
        answer = row["answer"]

        return (
            f'Given the question and context given tell me if the \n'
            f'answer is correct. Answer only with "True" or "False" \n'
            f"{context}\n"
            f'Answer: {answer}'
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "dataset_type": "local_json_one_per_line",
        "local_path": "/data/nfs-data/hotpot_qa_hallucinations_t5_xxl.json",
        "gt_processor": None,
        "input_processor": input_processor,
        "dataset_name": "hotpot_qa_hallucinated",
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # squad
    def input_processor(row):
        context = row["context"]
        question = row["question"]
        return (
            f'Using the context, answer the question with the exact wording '
            f'from the context.\nContext: {context}\n'
            f'Question: {question}'
        )

    def gt_processor(val):
        return val['text'][0]

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "answers"
        },
        "gt_processor": gt_processor,
        "input_processor": input_processor,
        "dataset_name": "squad",
        "hf_token": "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",
        "prompt": "",
        "split": "train",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # truthful_qa
    def input_processor(row):
        question = row["question"]
        shuffled_answers = []
        shuffled_answers.append(row["best_answer"])
        shuffled_answers.append(row["correct_answers"][0])
        try:
            shuffled_answers.append(row["correct_answers"][1])
            shuffled_answers.append(row["correct_answers"][2])
        except Exception as e:
            e
            pass
        shuffled_answers.append(row["incorrect_answers"][0])
        try:
            shuffled_answers.append(row["incorrect_answers"][1])
        except Exception as e:
            e
            pass
        random.shuffle(shuffled_answers)
        # dedupe answers
        shuffled_answers = list(dict.fromkeys(shuffled_answers))
        answers = ""
        for i in range(0, len(shuffled_answers)):
            answers += f"Answer {i+1}) {shuffled_answers[i]}\n"

        return (
            f"{question}\n"
            f'Choose the best answer and respond only with the full '
            "text of the answer: \n"
            f"{answers}"
        )

    cfg = {
        "column_map": {
            "input_column": "question",
            "gt_column": "best_answer"
        },
        "input_processor": input_processor,
        "dataset_name": "truthful_qa",
        "hf_token": "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",
        "prompt": "",
        "config_name": "generation",
        "split": "validation",
        "row_limit": row_limit,
    }

    # truthful_qa_multiple_exploded
    def input_processor(row):
        print(row)
        question = row["question"]
        answer = row["answer"]
        return (
            f'Tell me whether the answer to the question '
            "is correct or not: \n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "local_path": "/data/nfs-data/truthful_qa_exploded.json",
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "dataset_name": "truthful_qa_multiple_exploded",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # HALU_EVAL
    def input_processor_right_variant(row):
        document = row["document"]
        right_summary = row["right_summary"]

        return (
            'Respond only with one word. Based on the context given, '
            'tell me if the '
            'summary is correct or not. Output only True if it is correct '
            'and False if '
            "it's not. Do not include any other words in your output. \n"
            f"Summary: {right_summary}\n\n"
            f"Context: {document}"
        )

    def input_processor_hallucination_variant(row):
        document = row["document"]
        hallucinated_summary = row["hallucinated_summary"]

        return (
            "Respond only with one word. Based on the context given, "
            "tell me if the "
            "summary is correct or not. Output only True if it is "
            "correct and False if "
            "it's not. Do not include any other words in your output. \n"
            f"Summary: {hallucinated_summary}\n\n"
            f"Context: {document}"
        )

    def gt_processor_right_variant(val):
        return "True"

    def gt_processor_hallucination_variant(val):
        return "False"

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "local_path": "/data/nfs-data/halu_summary_data.json",
        "dataset_variant": dataset_variant,
        "dataset_name": "halu_eval_summary",
        "column_map": {
            "input_column": "document",
            "gt_column": "right_summary"
        },
        "input_processor": input_processor,
        "gt_processor": gt_processor,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }
    # Variants
    if dataset_variant is not None:
        if dataset_variant == "right":
            cfg["input_processor"] = input_processor_right_variant
            cfg["gt_processor"] = gt_processor_right_variant
        elif dataset_variant == "hallucination":
            cfg["input_processor"] = input_processor_hallucination_variant
            cfg["gt_processor"] = gt_processor_hallucination_variant
    else:
        # raise ValueError("Invalid dataset_variant. Variant required")
        pass

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # HALU_EVAL General Dataset
    def input_processor(row):
        question = row["user_query"]
        response = row["chatgpt_response"]

        return (
            'Based on the question given, tell me if the answer is correct  '
            'and accurate or not. Reply only with yes if it is correct and no '
            'if it\'s not\n'
            f"Question: {question}\n\n"
            f"Answer: {response}"
        )

    def gt_processor(val):
        if val == "yes":
            return "True"
        if val == "no":
            return "False"
        return val

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "halu_eval_general",
        "local_path": "/data/nfs-data/halu_general_data.json",
        "column_map": {
            "input_column": "user_query",
            "gt_column": "hallucination"
        },
        "input_processor": input_processor,
        "gt_processor": gt_processor,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # HALU_EVAL Q&A Dataset
    def input_processor_right_variant(row):
        knowledge = row["knowledge"]
        question = row["question"]
        right_answer = row["right_answer"]

        return (
            'Based on the context given, tell me if the answer to the '
            'question given is correct and accurate or not '
            'respond only with True if correct and False if not\n'
            f"Context: {knowledge}\n\n"
            f"Question: {question}\n\n"
            f"Answer: {right_answer}\n\n"
        )

    def input_processor_hallucination_variant(row):
        knowledge = row["knowledge"]
        question = row["question"]
        hallucinated_answer = row["hallucinated_answer"]

        return (
            'Based on the context given, tell me if the answer to the '
            'question given is correct and accurate or not '
            'respond only with True if correct and False if not\n'
            f"Context: {knowledge}\n\n"
            f"Question: {question}\n\n"
            f"Answer: {hallucinated_answer}\n\n"
        )

    def gt_processor_right_variant(val):
        return "True"

    def gt_processor_hallucination_variant(val):
        return "False"

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "local_path": "/data/nfs-data/halu_qa_data.json",
        "dataset_variant": dataset_variant,
        "dataset_name": "halu_eval_qa",
        "column_map": {
            "input_column": "knowledge",
            "gt_column": "right_answer"
        },
        "input_processor": None,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }
    # Variants
    if dataset_variant is not None:
        if dataset_variant == "right":
            cfg["input_processor"] = input_processor_right_variant
            cfg["gt_processor"] = gt_processor_right_variant
        elif dataset_variant == "hallucination":
            cfg["input_processor"] = input_processor_hallucination_variant
            cfg["gt_processor"] = gt_processor_hallucination_variant
    else:
        pass
        # raise ValueError("Invalid dataset_variant. Variant required")

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # HALU_EVAL Dialogue Dataset
    def input_processor_right_variant(row):
        knowledge = row["knowledge"]
        history = row["dialogue_history"]
        right_response = row["right_response"]

        return (
            'Output only one word - True or False.  '
            'Based on the knowledge and conversation history given, tell me '
            'if the response is correct and relevant or not. '
            'Respond only with True if correct and relevant and False if not\n'
            'Remember to output only one word.\n'
            f"Knowledge: {knowledge}\n\n"
            f"Conversation History: {history}\n\n"
            f"Response: {right_response}\n\n"
        )

    def input_processor_hallucination_variant(row):
        knowledge = row["knowledge"]
        history = row["dialogue_history"]
        hallucinated_response = row["hallucinated_response"]

        return (
            'Output only one word - True or False.  '
            'Based on the knowledge and conversation history given, tell me '
            'if the response is correct and relevant or not. '
            'Respond only with True if correct and relevant and False if not\n'
            'Remember to output only one word.\n'
            f"Knowledge: {knowledge}\n\n"
            f"Conversation History: {history}\n\n"
            f"Response: {hallucinated_response}\n\n"
        )

    def gt_processor_right_variant(val):
        return "True"

    def gt_processor_hallucination_variant(val):
        return "False"

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_variant": dataset_variant,
        "dataset_name": "halu_eval_dialogue",
        "local_path": "/data/nfs-data/halu_dialogue_data.json",
        "column_map": {
            "input_column": "knowledge",
            "gt_column": "right_response"
        },
        "input_processor": None,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }
    # Variants
    if dataset_variant is not None:
        if dataset_variant == "right":
            cfg["input_processor"] = input_processor_right_variant
            cfg["gt_processor"] = gt_processor_right_variant
        elif dataset_variant == "hallucination":
            cfg["input_processor"] = input_processor_hallucination_variant
            cfg["gt_processor"] = gt_processor_hallucination_variant
    else:
        pass
        # raise ValueError("Invalid dataset_variant. Variant required")

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # CAIS/MMLU - Law-Biz-Govt
    def input_processor(row):
        question = row["question"]
        answer = row["answer"]

        return (
            'Output only one word - True or False.  '
            'Based on the question given, tell me if the answer is correct  '
            'and accurate or not. Reply only with "True" if it is '
            'correct and "False" '
            'if it\'s not. Remember to output only one word.\n'
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "mmlu_law_biz_govt_right_wrong_repeated",
        "local_path":
            "//data/nfs-data/cais_mmlu_law_biz_govt_right_wrong_repeated.json",
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # CAIS/MMLU - Medicine
    def input_processor(row):
        question = row["question"]
        answer = row["answer"]

        return (
            'Output only one word - True or False.  '
            'Based on the question given, tell me if the answer is correct  '
            'and accurate or not. Reply only with "True" if it is '
            'correct and "False" '
            'if it\'s not. Remember to output only one word.\n'
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "mmlu_medicine_right_wrong_repeated",
        "local_path":
            "//data/nfs-data/cais_mmlu_medicine_right_wrong_repeated.json",
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # CAIS/MMLU - Science subjects
    def input_processor(row):
        question = row["question"]
        answer = row["answer"]

        return (
            'Output only one word - True or False.  '
            'Based on the question given, tell me if the answer is correct  '
            'and accurate or not. Reply only with "True" if it is '
            'correct and "False" '
            'if it\'s not. Remember to output only one word.\n'
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "mmlu_science_subjects_right_wrong_repeated",
        "local_path":
            "//data/nfs-data/cais_mmlu_science_subjects_"
            "right_wrong_repeated.json",
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # squad_v2 right/wrong answer pairs
    def input_processor(row):
        question = row["question"]
        answer = row["answer"]
        context = row["context"]

        return (
            'Output only one word - True or False.  '
            'Based on the question and context given, '
            'tell me if the answer is correct  '
            'and accurate or not. Reply only with "true" if it is '
            'correct and "false" '
            'if it\'s not. Remember to output only one word.\n'
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {context}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "squad_v2_right_wrong_pairs",
        "local_path": "//data/nfs-data/squad_v2_right_wrong.json",
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # squad_v2 right/wrong answer pairs with mistral generated wrong answers
    def input_processor(row):
        question = row["question"]
        answer = row["answer"]
        context = row["context"]

        return (
            'Output only one word - True or False.  '
            'Based on the question and context given, tell me if '
            'the answer is correct  '
            'and accurate or not. Reply only with "true" if it is '
            'correct and "false" '
            'if it\'s not. Remember to output only one word.\n'
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {context}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "squad_v2_right_wrong_pairs_mistral",
        "local_path": "//data/nfs-data/squad_v2_right_wrong-mistral.json",
        "column_map": {
            "input_column": "question",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # samsum right/wrong answer pairs
    def input_processor(row):
        dialogue = row["dialogue"]
        summary = row["summary"]

        return (
            'Output only one word - True or False.  '
            'Based on the dialogue and summary given, tell me if the summary '
            'is a correct and accurate summary of the dialogue. Reply only '
            'with "true" if it is correct and "false" if it\'s not\n'
            'Remember to output only one word.\n'
            f"Dialogue: {dialogue}\n"
            f"Summary: {summary}"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "samsum_right_wrong_pairs",
        "local_path": "//data/nfs-data/samsum_right_wrong.json",
        "column_map": {
            "input_column": "dialogue",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # allenai toxicity test
    def input_processor(row):
        text = row["prompt"]

        return (
            'Is the following text toxic? Reply only  '
            'with "true" if it is toxic and "false" if it\'s not\n'
            f"Text: {text}\n"
        )

    cfg = {
        "dataset_type": "local_json_one_per_line",
        "dataset_name": "allenai_toxicity",
        "local_path": "//data/nfs-data/allenai_toxicity.json",
        "column_map": {
            "input_column": "prompt",
            "gt_column": "ground_truth"
        },
        "input_processor": input_processor,
        "gt_processor": None,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # lmsys/mt_bench_human_judgments
    def input_processor(row):
        prompt = row['conversation_a'][0]['content']
        content_a = row['conversation_a'][1]['content']
        content_b = row['conversation_b'][1]['content']

        return (
            'Which response is better for the given prompt? '
            'Answer only with "A" or "B". \n'
            f'Prompt: {prompt}\n'
            f'Response A: {content_a}\n'
            f'Response B: {content_b}\n'
            'Remember to output only A or B.'
        )

    def gt_processor(val):
        if val == "model_a":
            return "0"
        if val == "model_b":
            return "1"
        if val == "tie":
            return "skip_row"

    cfg = {
        "dataset_name": "lmsys/mt_bench_human_judgments",
        "column_map": {
            "input_column": "",
            "gt_column": "winner"
        },
        "input_processor": input_processor,
        "gt_processor": gt_processor,
        "hf_token": "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",
        "prompt": "",
        "split": "human",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # diagram_hivemapper - /data/nfs-data/diagram/diagram_dataset.json
    def input_processor(row):
        context = row['question_and_context']
        answer = row['answer']

        return (
            'Based on the content and question given tell me if the '
            'answer to the question is correct.\n '
            'Answer only with a single word - yes or no. Remember to '
            'answer with only one word:\n'
            f'Answer: {answer}\n'
            f'{context}\n'
            'Remember to output only yes or no.'
        )

    def gt_processor(val):
        if val == "yes":
            return True
        if val == "no":
            return False
        return "skip_row"

    cfg = {
        "dataset_name": "diagram_hivemapper",
        "dataset_type": "local_json_one_per_line",
        "local_path": "/data/nfs-data/diagram/diagram_dataset.json",
        "column_map": {
            "input_column": "",
            "gt_column": "judgement"
        },
        "input_processor": input_processor,
        "gt_processor": gt_processor,
        "hf_token": "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",
        "prompt": "",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)

    # EleutherAI/race
    # https://huggingface.co/datasets/EleutherAI/race?row=0
    # Not working because of rogue single quotes
    """
    def single_processor(val):
        return val.replace("'", "")

    def input_processor(row):
        rp = row["problems"]
        rp = re.sub(r".*?'.*?", single_processor, rp)
        j = json.loads(
            rp
            .replace("'", '"')
            .replace("\\", ""))
        question = j[0]['question']
        answers = j[0]['options']
        context = row["article"]
        answers = (
                    f"A) {answers[0]} \nB) {answers[1]} \n"
                    f"C) {answers[2]} D) {answers[3]}"
                    )
        return (
            f"Choose the right answer to the question from the available "
            f"options based on the context. "
            f"Choose the right answer and output only the "
            f"letter of the answer and nothing else. \n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Possible answers: {answers} \n"
        )

    def gt_processor(val):
        return val[0]['answer']

    cfg = {
        "column_map": {
            "input_column": "article",
            "gt_column": "problems"
        },
        "gt_processor": gt_processor,
        "input_processor": input_processor,
        "dataset_name": "EleutherAI/race",
        "hf_token": "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",
        "prompt": "",
        "split": "test",
        "row_limit": row_limit,
    }

    add_to_dataset_dictionary(retVal, cfg, dataset_list, include)
    """

    return retVal
