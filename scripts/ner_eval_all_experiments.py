import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
from tqdm import tqdm
import numpy as np
from probe_mlp_layer import *


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def evaluate_model(model, dataloader, label_list):
    all_predictions = []
    all_true_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for i, label in enumerate(labels):
                true_labels = []
                pred_labels = []
                for j, m in enumerate(attention_mask[i]):
                    if m and labels[i][j] != -100:
                        true_labels.append(label_list[labels[i][j]])
                        pred_labels.append(label_list[predictions[i][j]])
                all_true_labels.append(true_labels)
                all_predictions.append(pred_labels)

    return f1_score(all_predictions, all_true_labels)


def _set_weight_param(model, attr_path: str, new_weight: torch.Tensor):
    """
    Dynamically set weight.data for a nested attribute path on model.
    attr_path: dot-separated, e.g. 'classifier.weight' or
               'distilbert.embeddings.position_embeddings.weight'
    """
    parts = attr_path.split('.')
    obj = model
    for p in parts[:-1]:
        # support numeric indices for lists
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    final = parts[-1]
    # set the underlying tensor data
    param = getattr(obj, final)
    param.data = new_weight.clone()


def run_weight_mod_sweep(
    model_name: str,
    W_orig: torch.Tensor,
    dataloader: DataLoader,
    label_list: list,
    noise_sigmas: np.ndarray,
    flip_fracs: np.ndarray,
    threshold_fracs: list,
    shuffle_types: list = ["random"],
    iters: int = 5,
    flip_method: str = "smallest",
    threshold_method: str = "smallest",
    weight_attr: str = 'classifier.weight'
) -> dict:
    """
    Run sweeps of noise, flip, threshold, and shuffle on a given weight matrix.

    Returns a dict with lists of results for each operation.
    {
      'baseline': float,
      'noise':     [{'sigma':..., 'flt_f1':..., 'bin_f1':...}, ...],
      'flip':      [{'q':...,     'flt_f1':..., 'bin_f1':...}, ...],
      'threshold': [{'frac':..., 'flt_f1':..., 'bin_f1':...}, ...],
      'shuffle':   [{'type':...,  'flt_f1':..., 'bin_f1':...}, ...],
    }
    """
    results = {
        'baseline': None,
        'noise': [],
        'flip': [],
        'threshold': [],
        'shuffle': []
    }

    # baseline
    base = AutoModelForTokenClassification.from_pretrained(model_name)
    results['baseline'] = evaluate_model(base, dataloader, label_list)

    for sigma in tqdm(noise_sigmas, desc='Noise injection'):
        flt_scores, bin_scores = [], []
        for _ in range(iters):
            # float weights with noise
            m = AutoModelForTokenClassification.from_pretrained(model_name)
            Wm_flt = modify_weights(
                W_orig,
                operation="noise",
                sigma=float(sigma),
                to_signs=False
            )
            _set_weight_param(m, weight_attr, Wm_flt)
            flt_scores.append(evaluate_model(m, dataloader, label_list))
            # binary weights via sign
            m = AutoModelForTokenClassification.from_pretrained(model_name)
            Wm_bin = torch.sign(Wm_flt)
            _set_weight_param(m, weight_attr, torch.sign(Wm_flt))
            bin_scores.append(evaluate_model(m, dataloader, label_list))
        res = {
            'sigma': float(sigma),
            'flt_f1': float(np.mean(flt_scores)),
            'bin_f1': float(np.mean(bin_scores))
        }
        results['noise'].append(res)

    for q in tqdm(flip_fracs, desc='Flipping weights'):
        res = {'q': float(q)}
        # float
        m = AutoModelForTokenClassification.from_pretrained(model_name)
        Wm_flt = modify_weights(W_orig, operation="flip", q=float(q), method=flip_method, to_signs=False)
        _set_weight_param(m, weight_attr, Wm_flt)
        res['flt_f1'] = evaluate_model(m, dataloader, label_list)
        # binary
        m = AutoModelForTokenClassification.from_pretrained(model_name)
        _set_weight_param(m, weight_attr, torch.sign(Wm_flt))
        res['bin_f1'] = evaluate_model(m, dataloader, label_list)
        results['flip'].append(res)

    for frac in tqdm(threshold_fracs, desc='Prunning'):
        res = {'frac': float(frac)}

        m = AutoModelForTokenClassification.from_pretrained(model_name)
        Wm_flt = modify_weights(
            W_orig,
            operation="threshold",
            method=threshold_method,
            fraction_non_zero=float(frac),
            to_signs=False
        )
        _set_weight_param(m, weight_attr, Wm_flt)
        res['flt_f1'] = evaluate_model(m, dataloader, label_list)

        m = AutoModelForTokenClassification.from_pretrained(model_name)
        _set_weight_param(m, weight_attr, torch.sign(Wm_flt))
        res['bin_f1'] = evaluate_model(m, dataloader, label_list)

        # Shuffle after prunning
        for stype in tqdm(shuffle_types, desc='Shuffling'):
            flt_scores, bin_scores = [], []
            for _ in range(iters):
                Wm_shuffle_flt = modify_weights(Wm_flt, operation="shuffle", shuffle_type=stype, is_sparse=True, to_signs=False)
                m = AutoModelForTokenClassification.from_pretrained(model_name)
                _set_weight_param(m, weight_attr, Wm_shuffle_flt)
                flt_scores.append(evaluate_model(m, dataloader, label_list)) 
                
                m = AutoModelForTokenClassification.from_pretrained(model_name)
                _set_weight_param(m, weight_attr, torch.sign(Wm_shuffle_flt))
                bin_scores.append(evaluate_model(m, dataloader, label_list))  

            res[f'shuffle_{stype}_flt_f1'] = float(np.mean(flt_scores))
            res[f'shuffle_{stype}_bin_f1'] = float(np.mean(bin_scores))

        results['threshold'].append(res)

    return results


def evaluate_ner_task():
    dataset = load_dataset("conll2003")

    model_name = "dslim/distilbert-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_datasets = dataset["test"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    columns_to_remove = ["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    eval_dataloader = DataLoader(
        tokenized_datasets, collate_fn=data_collator, batch_size=32
    )
    label_list = dataset["train"].features["ner_tags"].feature.names

    noise_sigmas    = np.linspace(0, 1, num=100)
    flip_fracs      = np.linspace(0, 1, num=50)
    threshold_fracs = [0.999,0.9,0.8,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,
                       0.3,0.25,0.2,0.15,0.1,0.05,0.04,0.03,0.02,0.01,
                       0.005,0.002,0.001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001]
    shuffle_types   = ["random", "random_pos_neg"]    
    iters = 5

    base = AutoModelForTokenClassification.from_pretrained(model_name)

    # Classifier layer
    W_cls = base.classifier.weight.data.clone()
    results_classifier = run_weight_mod_sweep(
        model_name,
        W_cls,
        eval_dataloader,
        label_list,
        noise_sigmas,
        flip_fracs,
        threshold_fracs,
        shuffle_types,
        flip_method="smallest",
        threshold_method="smallest",
        weight_attr='classifier.weight',
        iters=iters
    )

    # Positional encoding layer
    W_pos = base.distilbert.embeddings.position_embeddings.weight.data.clone()
    results_pos_enc = run_weight_mod_sweep(
        model_name, 
        W_pos, 
        eval_dataloader, 
        label_list,
        noise_sigmas,
        flip_fracs,
        threshold_fracs,
        shuffle_types,
        flip_method="smallest",
        threshold_method="smallest",
        weight_attr='distilbert.embeddings.position_embeddings.weight',
        iters=iters
    )

    # First transformer's MLP
    W_l1_first = base.distilbert.transformer.layer[0].ffn.lin1.weight.data.clone()
    results_l1_first = run_weight_mod_sweep(
        model_name, 
        W_l1_first, 
        eval_dataloader, 
        label_list,
        noise_sigmas,
        flip_fracs,
        threshold_fracs,
        shuffle_types,
        flip_method="smallest",
        threshold_method="smallest",
        weight_attr='distilbert.transformer.layer.0.ffn.lin1.weight',
        iters=iters
    )
    W_l2_first = base.distilbert.transformer.layer[0].ffn.lin2.weight.data.clone()
    results_l2_first = run_weight_mod_sweep(
        model_name, 
        W_l2_first, 
        eval_dataloader, 
        label_list,
        noise_sigmas,
        flip_fracs,
        threshold_fracs,
        shuffle_types,
        flip_method="smallest",
        threshold_method="smallest",
        weight_attr='distilbert.transformer.layer.0.ffn.lin2.weight',
        iters=iters
    )

    # Last transformer's MLP
    W_l1_last = base.distilbert.transformer.layer[5].ffn.lin1.weight.data.clone()
    results_l1_last = run_weight_mod_sweep(
        model_name, 
        W_l1_last, 
        eval_dataloader, 
        label_list,
        noise_sigmas,
        flip_fracs,
        threshold_fracs,
        shuffle_types,
        flip_method="smallest",
        threshold_method="smallest",
        weight_attr='distilbert.transformer.layer.5.ffn.lin1.weight',
        iters=iters
    )
    W_l2_last = base.distilbert.transformer.layer[5].ffn.lin2.weight.data.clone()
    results_l2_last = run_weight_mod_sweep(
        model_name, 
        W_l2_last, 
        eval_dataloader, 
        label_list,
        noise_sigmas,
        flip_fracs,
        threshold_fracs,
        shuffle_types,
        flip_method="smallest",
        threshold_method="smallest",
        weight_attr='distilbert.transformer.layer.5.ffn.lin2.weight',
        iters=iters
    )

    return results_classifier, results_pos_enc, results_l1_first, results_l2_first, results_l1_last, results_l2_last
