import torch
import os
import random
import utils_my
import data_utils
import argparse
import datetime
import json
from loguru import logger
import sys
import numpy as np
from collections import defaultdict
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from model.cbl import ConceptBottleneckLayer



import psutil
def print_mem_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)
    print(f"[MEM] {tag}: {mem:.3f} GB")


from torch.utils.data import Dataset
    
class MemmapConceptDataset(Dataset):
    def __init__(self, features, memmap_path, shape, dtype, shuffle=True):
        self.features = features
        self.memmap = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
        self.indices = np.arange(len(features))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        label = self.memmap[real_idx]
        label_tensor = torch.from_numpy(label.astype(np.float32))
        return self.features[real_idx], label_tensor

def create_concept_labels_memmap(features, images_to_class, concept_to_classes,
                                  new_text_features, out_path, Tconf, chunk_size=1024):
    dummy = utils_my.generate_concept_labels(features[:1], new_text_features, images_to_class[:1], concept_to_classes, Tconf)
    label_dim = dummy.shape[1]
    num_samples = len(features)
    memmap = np.memmap(out_path, dtype='float32', mode='w+', shape=(num_samples, label_dim))

    for i in range(0, num_samples, chunk_size):
        start, end = i, min(i + chunk_size, num_samples)
        chunk_labels = utils_my.generate_concept_labels(
            features[start:end], new_text_features,
            images_to_class[start:end], concept_to_classes, Tconf
        )
        memmap[start:end] = chunk_labels
        print(f"Concept labels chunk {i // chunk_size + 1} saved: {start}-{end}")

    memmap.flush()
    return memmap.shape, memmap.dtype


def compute_mean_std(cbl, features, batch_size, device):
    cbl.eval()
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    
    n = 0
    mean = 0
    M2 = 0
    
    with torch.no_grad():
        for (batch_feats,) in loader:
            batch_feats = batch_feats.to(device)
            batch_out = cbl(batch_feats).cpu()
            batch_size_ = batch_out.shape[0]
            batch_mean = batch_out.mean(dim=0)
            batch_var = batch_out.var(dim=0, unbiased=False)
            
            delta = batch_mean - mean
            total_n = n + batch_size_
            
            mean = (n * mean + batch_size_ * batch_mean) / total_n
            M2 = M2 + batch_size_ * batch_var + delta ** 2 * n * batch_size_ / total_n
            n = total_n
    
    std = torch.sqrt(M2 / n)
    return mean, std

class MemmapCBLDataset(Dataset):
    def __init__(self, memmap_path, shape, labels, dtype='float32', shuffle=True):
        self.memmap = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
        self.labels = labels
        self.indices = np.arange(len(self.memmap))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.memmap)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feat = torch.from_numpy(self.memmap[real_idx].astype(np.float32))
        label = self.labels[real_idx]
        return feat, label, real_idx


def save_cbl_features_to_memmap_norm(cbl, features, memmap_path, shape, mean, std, chunk_size=1024, dtype='float32', device='cuda'):
    cbl.eval()
    memmap = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)

    num_samples = features.shape[0]
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        batch_feats = features[start:end].to(device)
        with torch.no_grad():
            batch_out = cbl(batch_feats).cpu()
            batch_out = (batch_out - mean.cpu()) / std.cpu()
            batch_out = batch_out.numpy()

        memmap[start:end] = batch_out
        print(f"Saved normalized CBL features chunk: {start}-{end}")

    memmap.flush()
    return memmap


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def train_test_cbm_and_save(args):
    # Setup log directory and logger
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = f"{args.save_dir}/{args.dataset}/{args.dataset}_cbm_{timestamp}"
    while os.path.exists(save_dir):
        save_dir += "-1"
    os.makedirs(save_dir)
    
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info(f"Saving model to {save_dir}")
    
    # Load classes and concepts
    classes = data_utils.get_class_names(args.dataset)
    
    concept_set = args.concept_set or f"Concept generate/{args.dataset}_concept_class_wimage.json"
    with open(concept_set, 'r', encoding='utf-8') as f:
        concepts_data = json.load(f)
    concepts = concepts_data["concepts"]
    concepts_to_class = concepts_data["concepts_to_class"]

    # Prepare datasets
    d_train = f"{args.dataset}_train"
    d_val = f"{args.dataset}_val"
    
    # Get image-to-class mappings
    images_to_class_train = data_utils.get_targets_only(*d_train.split('_'))
    images_to_class_val = data_utils.get_targets_only(*d_val.split('_'))

    
    # Save activations for all datasets
    for d_probe in [d_train, d_val]:
        utils_my.save_all_features(
            clip_name=args.clip_name,
            target_name=args.backbone,
            target_layers=[args.feature_layer],
            d_probe=d_probe,
            concept_set_path=concept_set,
            batch_size=args.batch_size,
            device=args.device,
            pool_mode="avg",
            save_dir=args.activation_dir,
            use_penultimate=args.use_penultimate
        )


    # Load features with consistent naming
    def load_features(d_name):
        paths = utils_my.get_feature_paths(
            args.clip_name, 
            args.backbone, 
            d_name, 
            concept_set,
            [args.feature_layer] if args.feature_layer else None,  
            "avg", 
            args.activation_dir
        )
        print(paths)
        with torch.no_grad():
            if isinstance(paths["target"], dict):  
                target_feats = torch.load(paths["target"][args.feature_layer], map_location="cpu").float()
                #target_feats /= target_feats.norm(dim=1, keepdim=True)###
            else: 
                target_feats = torch.load(paths["target"], map_location="cpu").float()
                #target_feats /= target_feats.norm(dim=1, keepdim=True)###


            image_feats = torch.load(paths["clip"], map_location="cpu").float()
            image_feats = image_feats[::10]
            image_feats /= image_feats.norm(dim=1, keepdim=True)
            text_feats = torch.load(paths["text"], map_location="cpu").float()
            text_feats /= text_feats.norm(dim=1, keepdim=True)
            clip_feats = image_feats @ text_feats.T
            
        return target_feats, image_feats, text_feats, clip_feats

    # Load all features
    train_target, train_image, train_text, train_clip = load_features(d_train)
    val_target, val_image, val_text, val_clip = load_features(d_val)
    #test_target, test_image, _, _ = load_features(d_test)
    
    # Prepare targets (no splitting needed)
    train_targets = torch.LongTensor(images_to_class_train)
    val_targets = torch.LongTensor(images_to_class_val)
    #test_targets = torch.LongTensor(images_to_class_test)

    max_val = torch.max(train_clip).item()
    min_val = torch.min(train_clip).item()
    mean_val = torch.mean(train_clip).item()
    
    print(f"Max: {max_val:.4f}, Min: {min_val:.4f}, Mean: {mean_val:.4f}")

    # Filter and merge concepts (using CLIP features from train set)
    (new_concepts, 
     new_text_features, 
     concept_to_classes, 
     concept_redirect_map) = utils_my.filter_and_merge_concepts(
        train_clip, train_text, concepts, concepts_to_class,
        images_to_class_train[::10],  # Use full train set
        Tconf=args.Tconf, Tmerge=args.Tmerge, K=4, strategy="max",K_indep=args.K_indep
    )

    del train_image, train_clip, val_image, val_clip


    # Load features with consistent naming
    def load_features_new(d_name):
        paths = utils_my.get_feature_paths(
            args.clip_name, 
            args.backbone, 
            d_name, 
            concept_set,
            [args.feature_layer] if args.feature_layer else None,
            "avg", 
            args.activation_dir
        )
        with torch.no_grad():
            image_feats = torch.load(paths["clip"], map_location="cpu").float()
            image_feats /= image_feats.norm(dim=1, keepdim=True)

            
        return image_feats

      
    train_image = load_features_new(d_train)
    val_image = load_features_new(d_val)

    concept_labels_dir = '/root/autodl-tmp/imagenet_chunk/concept_labels/'
    os.makedirs(os.path.dirname(concept_labels_dir), exist_ok=True)
    concept_labels_train = concept_labels_dir+f"concept_labels_train_K{args.K_indep}.dat"
    concept_labels_val = concept_labels_dir+f"concept_labels_val_K{args.K_indep}.dat"
    
    if not os.path.exists(concept_labels_train):
        create_concept_labels_memmap(
            train_image, images_to_class_train, concept_to_classes, new_text_features,
            concept_labels_train, args.Tconf
        )
    else:
        print(f"Found {concept_labels_train}, skipping creation.")
    
    if not os.path.exists(concept_labels_val):
        create_concept_labels_memmap(
            val_image, images_to_class_val, concept_to_classes, new_text_features,
            concept_labels_val, args.Tconf
        )
    else:
        print(f"Found {concept_labels_val}, skipping creation.")


    train_shape = (len(train_image), len(new_concepts))
    val_shape = (len(val_image), len(new_concepts))
    del train_image, val_image
    
    train_concept_ds = MemmapConceptDataset(train_target, concept_labels_train, train_shape, 'float32')
    val_concept_ds = MemmapConceptDataset(val_target, concept_labels_val, val_shape, 'float32', shuffle=False)
    
    cbl_train_loader = DataLoader(train_concept_ds, batch_size=args.cbl_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    cbl_val_loader = DataLoader(val_concept_ds, batch_size=args.cbl_batch_size, shuffle=False)
    
    cbl = ConceptBottleneckLayer(
        input_dim=train_target.shape[1],
        concept_dim=train_shape[1],
        cbl_layer_num=args.cbl_layer_num,
        bias=args.cbl_bias
    ).to(args.device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cbl.parameters(), lr=args.cbl_lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for step in range(args.cbl_steps):
        print(step)
        cbl.train()
        for feats, labels in cbl_train_loader:
            feats = feats.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
    
            optimizer.zero_grad()
            outputs = cbl(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
        if step % 1 == 0 or step == args.cbl_steps - 1:
            cbl.eval()
            val_loss_sum = 0
            val_count = 0
            with torch.no_grad():
                for feats, labels in cbl_val_loader:
                    feats = feats.to(args.device)
                    labels = labels.to(args.device)
                    val_outputs = cbl(feats)
                    val_loss_sum += criterion(val_outputs, labels).item() * len(feats)
                    val_count += len(feats)
    
            val_loss = val_loss_sum / val_count
            logger.info(f"Step {step}: Val Loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = cbl.state_dict()
                #logger.info(f"Step {step}: Val Loss {val_loss:.4f}")
    
    cbl.load_state_dict(best_weights)

    num_samples = len(train_target)
    feature_dim = cbl(train_target[:1].to(args.device)).shape[1]

    memmap_dir = "/root/autodl-tmp/imagenet_chunk/cbl_logit"

    train_memmap_path = os.path.join(memmap_dir, f"cbl_logit_train_K{args.K_indep}.dat")
    val_memmap_path = os.path.join(memmap_dir, f"cbl_logit_val_K{args.K_indep}.dat")
    os.makedirs(os.path.dirname(train_memmap_path), exist_ok=True)
    

    train_mean, train_std = compute_mean_std(
        cbl, train_target,
        batch_size=args.cbl_batch_size,
        device=args.device
    )

    save_cbl_features_to_memmap_norm(
        cbl, train_target, train_memmap_path,
        (num_samples, feature_dim),
        train_mean, train_std,
        chunk_size=1024, device=args.device
    )

    save_cbl_features_to_memmap_norm(
        cbl, val_target, val_memmap_path,
        (len(val_target), feature_dim),
        train_mean, train_std,
        chunk_size=1024, device=args.device
    )
    
    train_cbl_ds = MemmapCBLDataset(train_memmap_path, (num_samples, feature_dim), labels=train_targets, shuffle=True)
    val_cbl_ds = MemmapCBLDataset(val_memmap_path, (len(val_target), feature_dim), labels=val_targets, shuffle=False)
    
    fl_train_loader = DataLoader(train_cbl_ds, batch_size=args.saga_batch_size, shuffle=True)
    fl_val_loader = DataLoader(val_cbl_ds, batch_size=args.saga_batch_size, shuffle=False)
        
    
    linear = torch.nn.Linear(len(new_concepts), len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}
    
    output_proj = glm_saga(
        linear,
        fl_train_loader,
        STEP_SIZE,
        args.n_iters,
        ALPHA,
        epsilon=1,
        k=1,
        val_loader=fl_val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=num_samples,
        n_classes=len(classes)
    )  # => end

    # Save additional human-readable files
    with open(os.path.join(save_dir, "concepts.txt"), 'w') as f:
        f.write("\n".join(new_concepts))
    
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Save all required components
    save_dict = {
        'cbl_state_dict': best_weights,
        'W_final_layer': output_proj['path'][0]['weight'],
        'b_final_layer': output_proj['path'][0]['bias'],
        'concepts': {
            'initial': concepts,
            'filtered': new_concepts,
            'concept_to_classes': concept_to_classes,
            'concept_redirect_map': concept_redirect_map
        }
    }
    torch.save(save_dict, os.path.join(save_dir, "full_model.pth"))

if __name__ == "__main__":
    sys.stdout = LoggerWriter("INFO")
    sys.stderr = LoggerWriter("DEBUG")

    parser = argparse.ArgumentParser(description="Train CBM model")
    parser.add_argument("--dataset", type=str, default="ImageNet")
    parser.add_argument("--concept_set", type=str, default="./data/generate_concept/concept/imagenet_concepts_gpt-4o_final.json", help="path to concept set name")
    parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

    parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("--saga_batch_size", type=int, default=512, help="Batch size used when fitting final layer")
    parser.add_argument("--cbl_batch_size", type=int, default=1024, help="Batch size to use when learning concept bottleneck layer")

    parser.add_argument("--feature_layer", type=str, default='layer4', 
                        help="Which layer to collect activations from. Should be the name of second to last layer in the model")
    parser.add_argument(
        "--use_penultimate", 
        action="store_true", 
        default=False,  
        help="Use penultimate layer"
    )
    parser.add_argument(
        "--K_indep",
        type=int,
        default=1, 
        help="Max number of independent components per class(default: 5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="adam weight_decay"
    )
    parser.add_argument("--activation_dir", type=str, default='/root/autodl-tmp/saved_activations8', help="save location for backbone and CLIP activations")
    parser.add_argument("--save_dir", type=str, default='imagenet_save_model', help="where to save trained models")
    parser.add_argument("--cbl_steps", type=int, default=60, help="max steps to train the concept bottleneck layer for")
    parser.add_argument("--cbl_lr", type=float, default=0.001, help="cbl_lr")
    parser.add_argument("--lam", type=float, default=0.0005, help="Sparsity regularization parameter, higher->more sparse")
    parser.add_argument("--n_iters", type=int, default=80, help="How many iterations to run the final layer solver for")
    parser.add_argument("--Tconf", type=float, default=0.20, help="Threshold for filtering and labeling")
    parser.add_argument("--Tmerge", type=float, default=0.9997, help="Threshold for deduplication and merging concepts")
    parser.add_argument("--cbl_layer_num", type=int, default=1, help="CBL layer num")
    parser.add_argument("--cbl_bias", action='store_true', help="Use bias in CBL layer")
    parser.add_argument("--seed", type=int, default=42, help="The random seed")


    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--config", type=str, default=None)
    config_arg, remaining_args = config_parser.parse_known_args()
    if config_arg.config is not None:
        with open(config_arg.config, "r") as f:
            config_arg = json.load(f)
        parser.set_defaults(**config_arg)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_test_cbm_and_save(args)