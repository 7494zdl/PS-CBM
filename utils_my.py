import os
import math
import torch
import clip
import json
import data_utils
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

PM_SUFFIX = {"max": "_max", "avg": ""}


def get_activation(outputs, mode):
    if mode == 'avg':
        def hook(model, input, output):
            if output.dim() == 4:
                outputs.append(output.mean(dim=[2, 3]).detach().cpu())
            else:
                outputs.append(output.detach().cpu())
    elif mode == 'max':
        def hook(model, input, output):
            if output.dim() == 4:
                outputs.append(output.amax(dim=[2, 3]).detach().cpu())
            else:
                outputs.append(output.detach().cpu())
    return hook


def _make_save_dir(save_name):
    save_dir = os.path.dirname(save_name)
    os.makedirs(save_dir, exist_ok=True)


def _all_saved(save_names):
    return all(os.path.exists(path) for path in save_names.values())


def save_clip_text_features(model, text, save_path, batch_size=1000):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text) / batch_size))):
            features.append(model.encode_text(text[i * batch_size:(i + 1) * batch_size]))
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()


def save_clip_image_features(model, dataset, save_path, batch_size=1000, device="cuda"):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features.append(model.encode_image(images.to(device)).cpu())
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()


def save_clip_rn_penultimate_features(model, dataset, save_path, batch_size=1000, device="cuda"):
    if os.path.exists(save_path): return
    _make_save_dir(save_path)
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features.append(model(images.to(device)).cpu())
    torch.save(torch.cat(features), save_path)
    torch.cuda.empty_cache()
    

def save_target_activations(model, dataset, save_template, target_layers, batch_size=1000, device="cuda", pool_mode="avg"):
    
    save_paths = {layer: save_template.format(layer) for layer in target_layers}
    if _all_saved(save_paths): 
        return

    outputs = {layer: [] for layer in target_layers}
    hooks = {}

    for layer in target_layers:
        try:
            layer_obj = _get_layer_by_name(model, layer)

            hooks[layer] = layer_obj.register_forward_hook(
                get_activation(outputs[layer], pool_mode)
            )
        except AttributeError as e:
            raise ValueError(f"No such '{layer}': {str(e)}")

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            model(images.to(device))

    for layer in target_layers:
        torch.save(torch.cat(outputs[layer]), save_paths[layer])
        hooks[layer].remove() 
    
    torch.cuda.empty_cache()

def _get_layer_by_name(model, layer_name):

    parts = layer_name.split('.')
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current

def get_feature_paths(clip_name, target_name, d_probe, concept_set_path, target_layers=None, pool_mode=None, save_dir="."):
   
    clip_feat_path = f"{save_dir}/{d_probe}_clip_{clip_name.replace('/', '')}.pt"
    
    concept_name = os.path.splitext(os.path.basename(concept_set_path))[0]
    text_feat_path = f"{save_dir}/{concept_name}_{clip_name.replace('/', '')}.pt"
    
    if target_name.startswith("clip_"):
        target_paths = f"{save_dir}/{d_probe}_{target_name[5:].replace('/', '')}.pt"
    else:
        if target_layers is None or pool_mode is None:
            raise ValueError("Error!")
        target_paths = {
            layer: f"{save_dir}/{d_probe}_{target_name}_{layer}{PM_SUFFIX[pool_mode]}.pt"
            for layer in target_layers
        }
    
    return {
        "clip": clip_feat_path,
        "text": text_feat_path,
        "target": target_paths
    }

def save_all_features(clip_name, target_name, target_layers, d_probe, concept_set_path,
                    batch_size, device, pool_mode, save_dir, use_penultimate=False):

    paths = get_feature_paths(
        clip_name=clip_name,
        target_name=target_name,
        d_probe=d_probe,
        concept_set_path=concept_set_path,
        target_layers=target_layers,
        pool_mode=pool_mode,
        save_dir=save_dir
    )
    

    check_paths = {"clip": paths["clip"], "text": paths["text"]}
    if isinstance(paths["target"], dict):
        check_paths.update(paths["target"])
    else:
        check_paths["target"] = paths["target"]
    
    if _all_saved(check_paths): 
        return

    clip_model, clip_preprocess = clip.load(clip_name, device=device)

    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)

    dataset_name, split = d_probe.split("_")
    data_c = data_utils.get_data(dataset_name, split, clip_preprocess)
    data_t = data_utils.get_data(dataset_name, split, target_preprocess)

    with open(concept_set_path, 'r', encoding='utf-8') as f:
        concepts = json.load(f)["concepts"]
    text_tokens = clip.tokenize(concepts).to(device)

    save_clip_text_features(clip_model, text_tokens, paths["text"], batch_size)
    save_clip_image_features(clip_model, data_c, paths["clip"], batch_size, device)

    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
        visual = target_model.visual
        
        if use_penultimate:
            if hasattr(visual, "attnpool") and hasattr(visual.attnpool, "c_proj"):
                N = visual.attnpool.c_proj.in_features
                identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
                torch.nn.init.zeros_(identity.bias)
                identity.weight.data.copy_(torch.eye(N))
                visual.attnpool.c_proj = identity
            elif hasattr(visual, "proj"):
                if isinstance(visual.proj, torch.nn.Parameter):
                    N = visual.proj.shape[0]
                    visual.proj = torch.nn.Parameter(torch.eye(N, dtype=torch.float16, device=device))
                else:
                    N = visual.proj.in_features
                    identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
                    torch.nn.init.zeros_(identity.bias)
                    identity.weight.data.copy_(torch.eye(N))
                    visual.proj = identity
            else:
                raise ValueError(f"Unknown CLIP visual model: {target_name}")
            save_clip_rn_penultimate_features(visual.float(), data_t, paths["target"], batch_size, device)
        else: 
            save_clip_image_features(target_model, data_t, paths["target"], batch_size, device)
    
    else:
        save_target_activations(
            target_model, 
            data_t, 
            f"{save_dir}/{d_probe}_{target_name}_" + "{}.pt",
            target_layers, 
            batch_size, 
            device, 
            pool_mode
        )



def filter_and_merge_concepts(
    clip_features, text_features, concepts, concepts_to_class,
    images_to_class_train, Tconf=0.5, Tmerge=0.95, K=4, strategy='max',
    K_indep=5
):
    """
    Step 1: Filter weak concepts by Tconf
    Step 2: Merge similar concepts by Tmerge using greedy coverage
    Step 3: Prune redundant independent concepts per class by top-K mean similarity

    strategy: 'max' or 'median'
        - 'max': select concept with largest coverage set
        - 'median': select concept whose coverage size is median among uncovered
    """
    n_images, n_concepts = clip_features.shape

    # Step 1: Filter weak concepts
    valid_idx = []
    for i, cls in enumerate(concepts_to_class):
        img_idxs = [j for j, c in enumerate(images_to_class_train) if c == cls]
        if len(img_idxs) == 0:
            continue
        
        sims = clip_features[img_idxs, i]
        topk = sims.topk(min(K, sims.shape[0])).values
        mean_topk = topk.mean().item()
        if mean_topk >= Tconf:
            valid_idx.append(i)

    print(f"Initial concepts before filtering: {len(concepts)}")
    # Filter data
    concepts = [concepts[i] for i in valid_idx]
    concepts_to_class = [concepts_to_class[i] for i in valid_idx]
    text_features = text_features[valid_idx]
    clip_features = clip_features[:, valid_idx]
    
    print(f"Remaining concepts after filtering: {len(valid_idx)}")

    # Step 2: Merge similar concepts
    concept_vecs = clip_features.T
    concept_vecs = concept_vecs / concept_vecs.norm(dim=1, keepdim=True)
    sim_matrix = concept_vecs @ concept_vecs.T

    covered_sets = [
        set((sim_matrix[i] >= Tmerge).nonzero(as_tuple=True)[0].tolist())
        for i in range(len(concepts))
    ]
    uncovered = set(range(len(concepts)))
    representative_of = {}
    representatives = []

    while uncovered:
        cover_sizes = [(i, len(covered_sets[i] & uncovered)) for i in uncovered]

        if strategy == 'max':
            best = max(cover_sizes, key=lambda x: x[1])[0]
        elif strategy == 'median':
            sorted_sizes = sorted(size for _, size in cover_sizes)
            median_size = sorted_sizes[len(sorted_sizes) // 2]
            median_candidates = [i for i, size in cover_sizes if size == median_size]
            best = min(median_candidates)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        covers = covered_sets[best] & uncovered
        for c in covers:
            representative_of[c] = best
        representatives.append(best)
        uncovered -= covers

    # Remap indices
    representatives = sorted(set(representatives))
    old_to_new = {old: new_idx for new_idx, old in enumerate(representatives)}
    concept_redirect_map = {
        old_idx: old_to_new[representative_of[old_idx]]
        for old_idx in range(len(concepts))
    }

    # Build structures
    merged_concepts = [concepts[i] for i in representatives]
    merged_text_features = text_features[representatives]

    concept_to_classes = defaultdict(set)
    for old_idx, cls in enumerate(concepts_to_class):
        new_idx = concept_redirect_map[old_idx]
        concept_to_classes[new_idx].add(cls)

    print(f"Remaining concepts after merging: {len(merged_concepts)}")

    # Step 3: For each class, keep top-K independent concepts
    class_to_concepts = defaultdict(list)
    for concept_idx, class_set in concept_to_classes.items():
        for cls in class_set:
            class_to_concepts[cls].append(concept_idx)

    final_representatives = set()

    for cls, concept_idxs in class_to_concepts.items():
        shared = [i for i in concept_idxs if len(concept_to_classes[i]) > 1]
        unique = [i for i in concept_idxs if len(concept_to_classes[i]) == 1]

        img_idxs = [j for j, c in enumerate(images_to_class_train) if c == cls]
        if not img_idxs:
            continue
        image_features = clip_features[img_idxs][:, representatives]  # [N, M]

        concept_scores = []
        for i in unique:
            sim_scores = image_features[:, i]
            mean_score = sim_scores.mean().item()
            concept_scores.append((i, mean_score))

        concept_scores.sort(key=lambda x: -x[1])
        topk_unique = [i for i, _ in concept_scores[:K_indep]]

        final_representatives.update(shared)
        final_representatives.update(topk_unique)

    # Final mapping
    final_representatives = sorted(final_representatives)
    old_to_new_final = {old: i for i, old in enumerate(final_representatives)}

    new_concepts = [merged_concepts[i] for i in final_representatives]
    new_text_features = merged_text_features[final_representatives]

    concept_to_classes_final = {
        old_to_new_final[i]: concept_to_classes[i]
        for i in final_representatives
    }

    concept_redirect_map_final = {}
    for old_idx in range(len(concepts_to_class)):
        rep_idx = representative_of[old_idx]
        if rep_idx in representatives and old_to_new.get(rep_idx) in old_to_new_final:
            concept_redirect_map_final[old_idx] = old_to_new_final[old_to_new[rep_idx]]

    print(f"Remaining concepts after pruning: {len(new_concepts)}")
    print(new_concepts)

    return (
        new_concepts,
        new_text_features,
        concept_to_classes_final,
        concept_redirect_map_final,
    )


def generate_concept_labels(image_features, text_features, images_to_class, concept_to_classes, Tconf=0.9):
    """
    Generate multi-hot label vectors for each image
    """
    sim_matrix = image_features @ text_features.T  # [n_images, n_concepts]
    concept_labels = torch.zeros((image_features.size(0), text_features.size(0)))

    for concept_idx in range(text_features.size(0)):
        valid_classes = concept_to_classes[concept_idx]
        for img_idx in range(image_features.size(0)):
            if images_to_class[img_idx] in valid_classes and sim_matrix[img_idx, concept_idx] >= Tconf:
                concept_labels[img_idx, concept_idx] = 1.0

    return concept_labels
