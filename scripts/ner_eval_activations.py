import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from probe_mlp_layer import modify_weights  # your existing implementation


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


########################################
# Data preparation (same as your code)
########################################

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


def build_ner_dataloader(model_name: str, split: str = "test", batch_size: int = 32):
    dataset = load_dataset("conll2003")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized = dataset[split].map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    # remove original label/token fields
    columns_to_remove = ["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"]
    tokenized = tokenized.remove_columns(columns_to_remove)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = DataLoader(
        tokenized,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader


########################################
# Module access and hooks
########################################

def get_module_by_path(model, module_path: str):
    """
    Traverse a nested attribute path like
    'distilbert.transformer.layer.0.ffn.lin1'
    and return that module.
    """
    parts = module_path.split(".")
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def collect_activations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_paths: dict,
    device: torch.device,
    max_batches: int = 10,
    max_samples: int = 5000,
):
    logger.info(
        "Collecting activations: layers=%s, max_batches=%d, max_samples=%d",
        list(layer_paths.keys()), max_batches, max_samples
    )

    model.to(device)
    model.eval()

    activations = {name: [] for name in layer_paths.keys()}
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            out = output
            if isinstance(out, tuple):
                out = out[0]
            out = out.detach().to("cpu")
            if out.dim() == 3:
                # (batch, seq, dim) -> (batch*seq, dim)
                out = out.view(out.size(0) * out.size(1), -1)
            elif out.dim() == 2:
                out = out.view(out.size(0), -1)
            else:
                out = out.view(out.size(0), -1)
            activations[layer_name].append(out)
        return hook

    for name, path in layer_paths.items():
        module = get_module_by_path(model, path)
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                logger.info("Reached max_batches=%d while collecting activations", max_batches)
                break
            if i % 2 == 0:
                logger.info("Activation collection: batch %d", i)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    for name in activations:
        if len(activations[name]) == 0:
            logger.warning("No activations collected for layer '%s'", name)
            activations[name] = torch.empty(0)
            continue

        acts = torch.cat(activations[name], dim=0)

        # IMPORTANT CHANGE: deterministic, shared subset instead of random subset
        if acts.size(0) > max_samples:
            logger.info(
                "Truncating activations for '%s' from %d to %d (first samples)",
                name, acts.size(0), max_samples
            )
            acts = acts[:max_samples]

        activations[name] = acts

    logger.info("Finished collecting activations.")
    return activations


########################################
# Linear CKA implementation
########################################

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two activation matrices X, Y of shape (N, D).
    Implementation in torch, returns a Python float.
    """
    # center features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    K = X @ X.t()
    L = Y @ Y.t()

    # center Gram matrices
    n = K.size(0)
    H = torch.eye(n) - torch.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    # HSIC and normalisation
    hsic = (Kc * Lc).sum()
    norm_x = torch.sqrt((Kc * Kc).sum())
    norm_y = torch.sqrt((Lc * Lc).sum())
    cka = hsic / (norm_x * norm_y + 1e-12)
    return cka.item()


########################################
# Weight setters / probes
########################################

def set_weight_param(model, attr_path: str, new_weight: torch.Tensor):
    """
    Set weight.data at a nested attribute path (e.g. 'classifier.weight').
    """
    parts = attr_path.split(".")
    obj = model
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    final = parts[-1]
    param = getattr(obj, final)
    param.data = new_weight.clone().to(param.data.device)


def run_cka_sweep_for_layer(
    model_name: str,
    dataloader: DataLoader,
    layer_paths: dict,
    weight_attr: str,
    W_orig: torch.Tensor,
    noise_sigmas: np.ndarray = None,
    flip_fracs: np.ndarray = None,
    threshold_fracs: np.ndarray = None,
    shuffle_types: list = None,
    max_batches: int = 10,
    max_samples: int = 5000,
    iters: int = 1,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Starting CKA sweep for weight_attr='%s'. "
        "noise_sigmas=%s, flip_fracs=%s, threshold_fracs=%s, shuffle_types=%s, iters=%d",
        weight_attr,
        None if noise_sigmas is None else list(np.round(noise_sigmas, 3)),
        None if flip_fracs is None else list(np.round(flip_fracs, 3)),
        None if threshold_fracs is None else list(np.round(threshold_fracs, 3)),
        shuffle_types,
        iters,
    )

    results = {"noise": [], "flip": [], "threshold": []}

    logger.info("Computing baseline activations (unperturbed model).")
    base_model = AutoModelForTokenClassification.from_pretrained(model_name)
    base_acts = collect_activations(
        base_model, dataloader, layer_paths, device,
        max_batches=max_batches, max_samples=max_samples
    )

    # --- Noise ---
    if noise_sigmas is not None:
        logger.info("Running noise sweep (%d values).", len(noise_sigmas))
        for idx, sigma in enumerate(noise_sigmas):
            logger.info("Noise step %d/%d: sigma=%.4f", idx + 1, len(noise_sigmas), sigma)
            cka_per_layer = {name: [] for name in layer_paths.keys()}
            for it in range(iters):
                logger.info("  Noise replicate %d/%d", it + 1, iters)
                m = AutoModelForTokenClassification.from_pretrained(model_name)
                W_noise = modify_weights(
                    W_orig,
                    operation="noise",
                    sigma=float(sigma),
                    to_signs=False
                )
                set_weight_param(m, weight_attr, W_noise)
                pert_acts = collect_activations(
                    m, dataloader, layer_paths, device,
                    max_batches=max_batches, max_samples=max_samples
                )
                for lname in layer_paths.keys():
                    X = base_acts[lname]
                    Y = pert_acts[lname]
                    if X.numel() == 0 or Y.numel() == 0:
                        logger.warning("Empty activations for layer '%s' under noise sigma=%.4f", lname, sigma)
                        continue
                    cka_val = linear_cka(X, Y)
                    cka_per_layer[lname].append(cka_val)
            cka_mean = {
                k: float(np.mean(v)) if len(v) > 0 else float("nan")
                for k, v in cka_per_layer.items()
            }
            results["noise"].append({"sigma": float(sigma), "cka": cka_mean})

    # --- Flip ---
    if flip_fracs is not None:
        logger.info("Running sign-flip sweep (%d values).", len(flip_fracs))
        for idx, q in enumerate(flip_fracs):
            logger.info("Flip step %d/%d: q=%.4f", idx + 1, len(flip_fracs), q)
            cka_per_layer = {name: [] for name in layer_paths.keys()}
            for it in range(iters):
                logger.info("  Flip replicate %d/%d", it + 1, iters)
                m = AutoModelForTokenClassification.from_pretrained(model_name)
                W_flip = modify_weights(
                    W_orig,
                    operation="flip",
                    q=float(q),
                    method="smallest",
                    to_signs=False
                )
                set_weight_param(m, weight_attr, W_flip)
                pert_acts = collect_activations(
                    m, dataloader, layer_paths, device,
                    max_batches=max_batches, max_samples=max_samples
                )
                for lname in layer_paths.keys():
                    X = base_acts[lname]
                    Y = pert_acts[lname]
                    if X.numel() == 0 or Y.numel() == 0:
                        logger.warning("Empty activations for layer '%s' under flip q=%.4f", lname, q)
                        continue
                    cka_val = linear_cka(X, Y)
                    cka_per_layer[lname].append(cka_val)
            cka_mean = {
                k: float(np.mean(v)) if len(v) > 0 else float("nan")
                for k, v in cka_per_layer.items()
            }
            results["flip"].append({"q": float(q), "cka": cka_mean})

    # --- Threshold + shuffle ---
    if threshold_fracs is not None:
        if shuffle_types is None:
            shuffle_types = []
        logger.info("Running threshold sweep (%d values).", len(threshold_fracs))
        for idx, frac in enumerate(threshold_fracs):
            logger.info("Threshold step %d/%d: frac=%.6f", idx + 1, len(threshold_fracs), frac)
            entry = {"frac": float(frac), "cka": {}}

            # Threshold only
            cka_per_layer = {name: [] for name in layer_paths.keys()}
            for it in range(iters):
                logger.info("  Threshold replicate %d/%d", it + 1, iters)
                m = AutoModelForTokenClassification.from_pretrained(model_name)
                W_thr = modify_weights(
                    W_orig,
                    operation="threshold",
                    method="smallest",
                    fraction_non_zero=float(frac),
                    to_signs=False
                )
                set_weight_param(m, weight_attr, W_thr)
                pert_acts = collect_activations(
                    m, dataloader, layer_paths, device,
                    max_batches=max_batches, max_samples=max_samples
                )
                for lname in layer_paths.keys():
                    X = base_acts[lname]
                    Y = pert_acts[lname]
                    if X.numel() == 0 or Y.numel() == 0:
                        logger.warning("Empty activations for layer '%s' under threshold frac=%.6f", lname, frac)
                        continue
                    cka_val = linear_cka(X, Y)
                    cka_per_layer[lname].append(cka_val)
            entry["cka"] = {
                k: float(np.mean(v)) if len(v) > 0 else float("nan")
                for k, v in cka_per_layer.items()
            }

            # Threshold + shuffle variants
            for stype in shuffle_types:
                logger.info("  Threshold+shuffle type='%s' for frac=%.6f", stype, frac)
                cka_per_layer = {name: [] for name in layer_paths.keys()}
                for it in range(iters):
                    logger.info("    Shuffle replicate %d/%d", it + 1, iters)
                    W_thr = modify_weights(
                        W_orig,
                        operation="threshold",
                        method="smallest",
                        fraction_non_zero=float(frac),
                        to_signs=False
                    )
                    W_shuf = modify_weights(
                        W_thr,
                        operation="shuffle",
                        shuffle_type=stype,
                        is_sparse=True,
                        to_signs=False
                    )
                    m = AutoModelForTokenClassification.from_pretrained(model_name)
                    set_weight_param(m, weight_attr, W_shuf)
                    pert_acts = collect_activations(
                        m, dataloader, layer_paths, device,
                        max_batches=max_batches, max_samples=max_samples
                    )
                    for lname in layer_paths.keys():
                        X = base_acts[lname]
                        Y = pert_acts[lname]
                        if X.numel() == 0 or Y.numel() == 0:
                            logger.warning(
                                "Empty activations for '%s' under threshold+shuffle type='%s', frac=%.6f",
                                lname, stype, frac
                            )
                            continue
                        cka_val = linear_cka(X, Y)
                        cka_per_layer[lname].append(cka_val)
                entry[f"cka_shuffle_{stype}"] = {
                    k: float(np.mean(v)) if len(v) > 0 else float("nan")
                    for k, v in cka_per_layer.items()
                }

            results["threshold"].append(entry)

    logger.info("Finished CKA sweep for weight_attr='%s'.", weight_attr)
    return results



########################################
# Example main: 6 layers, small sweeps
########################################

def main():
    model_name = "dslim/distilbert-NER"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataloader = build_ner_dataloader(model_name, split="test", batch_size=16)
    logger.info("Built NER dataloader.")

    layer_paths = {
        "pos_enc": "distilbert.embeddings.position_embeddings",
        "ffn0_lin1": "distilbert.transformer.layer.0.ffn.lin1",
        "ffn0_lin2": "distilbert.transformer.layer.0.ffn.lin2",
        "ffn5_lin1": "distilbert.transformer.layer.5.ffn.lin1",
        "ffn5_lin2": "distilbert.transformer.layer.5.ffn.lin2",
        "classifier": "classifier",
    }

    weight_specs = {
        "classifier": "classifier.weight",
        "pos_enc": "distilbert.embeddings.position_embeddings.weight",
        "ffn0_lin1": "distilbert.transformer.layer.0.ffn.lin1.weight",
        "ffn0_lin2": "distilbert.transformer.layer.0.ffn.lin2.weight",
        "ffn5_lin1": "distilbert.transformer.layer.5.ffn.lin1.weight",
        "ffn5_lin2": "distilbert.transformer.layer.5.ffn.lin2.weight",
    }

    logger.info("Loading base model '%s' to extract original weights.", model_name)
    base = AutoModelForTokenClassification.from_pretrained(model_name)

    noise_sigmas = np.linspace(0.0, 1, num=20)
    flip_fracs = np.linspace(0.0, 1, num=20)
    threshold_fracs = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])

    shuffle_types = ["random", "random_pos_neg"]
    max_batches = 64
    max_samples = 5000
    iters = 3

    all_results = {}

    for key, wpath in weight_specs.items():
        logger.info("=== Starting layer '%s' (%s) ===", key, wpath)
        module = get_module_by_path(base, ".".join(wpath.split(".")[:-1]))
        W_orig = module.weight.data.clone()

        res = run_cka_sweep_for_layer(
            model_name=model_name,
            dataloader=dataloader,
            layer_paths=layer_paths,
            weight_attr=wpath,
            W_orig=W_orig,
            noise_sigmas=noise_sigmas,
            flip_fracs=flip_fracs,
            threshold_fracs=threshold_fracs,
            shuffle_types=shuffle_types,
            max_batches=max_batches,
            max_samples=max_samples,
            iters=iters,
            device=device,
        )
        
        print(key, wpath, res)

        all_results[key] = res
        logger.info("=== Finished layer '%s' ===", key)

    logger.info("All CKA sweeps completed. Ready to save results.")

    import json
    with open("cka_results_fixed_29_11_25.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
