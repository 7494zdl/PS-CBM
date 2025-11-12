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
    d_test = f"{args.dataset}_test"
    
    # Get image-to-class mappings
    images_to_class_train = data_utils.get_targets_only(*d_train.split('_'))
    images_to_class_val = data_utils.get_targets_only(*d_val.split('_'))
    images_to_class_test = data_utils.get_targets_only(*d_test.split('_'))

    # Save activations for all datasets
    for d_probe in [d_train, d_val, d_test]:
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
        
        with torch.no_grad():

            if isinstance(paths["target"], dict): 
                target_feats = torch.load(paths["target"][args.feature_layer], map_location="cpu").float()
                #target_feats /= target_feats.norm(dim=1, keepdim=True)###
            else: 
                target_feats = torch.load(paths["target"], map_location="cpu").float()
                #target_feats /= target_feats.norm(dim=1, keepdim=True)###
                
            image_feats = torch.load(paths["clip"], map_location="cpu").float()
            image_feats /= image_feats.norm(dim=1, keepdim=True)
            text_feats = torch.load(paths["text"], map_location="cpu").float()
            text_feats /= text_feats.norm(dim=1, keepdim=True)
            clip_feats = image_feats @ text_feats.T
            
        return target_feats, image_feats, text_feats, clip_feats

    # Load all features
    train_target, train_image, train_text, train_clip = load_features(d_train)
    val_target, val_image, val_text, val_clip = load_features(d_val)
    test_target, test_image, _, _ = load_features(d_test)
    
    # Prepare targets (no splitting needed)
    train_targets = torch.LongTensor(images_to_class_train)
    val_targets = torch.LongTensor(images_to_class_val)
    test_targets = torch.LongTensor(images_to_class_test)

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
        images_to_class_train,  # Use full train set
        Tconf=args.Tconf, Tmerge=args.Tmerge, K=4, strategy="max",K_indep=args.K_indep
    )

    # Generate concept labels for all datasets
    concept_labels_train = utils_my.generate_concept_labels(
        train_image, new_text_features, 
        images_to_class_train, concept_to_classes, Tconf=args.Tconf
    )
    concept_labels_val = utils_my.generate_concept_labels(
        val_image, new_text_features,
        images_to_class_val, concept_to_classes, Tconf=args.Tconf
    )
    concept_labels_test = utils_my.generate_concept_labels(
        test_image, new_text_features,
        images_to_class_test, concept_to_classes, Tconf=args.Tconf
    )


    # Train Concept Bottleneck Layer (using full train set)
    cbl = ConceptBottleneckLayer(
        input_dim=train_target.shape[1],
        concept_dim=len(new_concepts),
        cbl_layer_num=args.cbl_layer_num,
        bias=args.cbl_bias
    ).to(args.device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cbl.parameters(), lr=args.cbl_lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for step in range(args.cbl_steps):
        # Mini-batch training (full train set)
        idx = torch.randperm(len(train_target))[:args.cbl_batch_size]
        feats = train_target[idx].to(args.device)
        labels = concept_labels_train[idx].to(args.device)
        
        optimizer.zero_grad()
        outputs = cbl(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Validation (full val set)
        if step % 50 == 0 or step == args.cbl_steps - 1:
            with torch.no_grad():
                val_outputs = cbl(val_target.to(args.device))
                val_loss = criterion(val_outputs, concept_labels_val.to(args.device))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = cbl.state_dict()
                    logger.info(f"Step {step}: Train Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}")

    # Load best weights
    cbl.load_state_dict(best_weights)
    
    # Extract and normalize features
    with torch.no_grad():
        # Train features
        train_c = cbl(train_target.to(args.device))
        train_mean = train_c.mean(dim=0, keepdim=True)
        train_std = train_c.std(dim=0, keepdim=True)
        train_c = (train_c - train_mean) / train_std
        
        # Val features
        val_c = cbl(val_target.to(args.device))
        val_c = (val_c - train_mean) / train_std
        
        # Test features
        test_c = cbl(test_target.to(args.device))
        test_c = (test_c - train_mean) / train_std

    # Prepare datasets
    #train_ds = TensorDataset(train_c.cpu(), train_targets)
    #val_ds = TensorDataset(val_c.cpu(), val_targets)
    test_ds = TensorDataset(test_c.cpu(), test_targets)

    train_ds = IndexedTensorDataset(train_c.cpu(), train_targets)
    val_ds = IndexedTensorDataset(val_c.cpu(), val_targets)

    # Train final layer
    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}
    output_proj = glm_saga(
        linear, 
        DataLoader(train_ds, batch_size=args.saga_batch_size, shuffle=True),
        STEP_SIZE,
        args.n_iters,
        ALPHA,
        epsilon=1, 
        k=1,
        val_loader=DataLoader(val_ds, batch_size=args.saga_batch_size),
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_ds),
        n_classes=len(classes)
    )

    # Save all required components
    save_dict = {
        'cbl_state_dict': best_weights,
        'W_final_layer': output_proj['path'][0]['weight'],
        'b_final_layer': output_proj['path'][0]['bias'],
        'normalization': {
            'train_mean': train_mean.cpu(),
            'train_std': train_std.cpu()
        },
        'concepts': {
            'initial': concepts,
            'filtered': new_concepts,
            'concept_to_classes': concept_to_classes,
            'concept_redirect_map': concept_redirect_map
        },
        'args': vars(args),
        'concept_labels': {
            'train': concept_labels_train.cpu(),
            'val': concept_labels_val.cpu(),
            'test': concept_labels_test.cpu()
        }
    }
    torch.save(save_dict, os.path.join(save_dir, "full_model.pth"))

    # Save additional human-readable files
    with open(os.path.join(save_dir, "concepts.txt"), 'w') as f:
        f.write("\n".join(new_concepts))
    
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Test evaluation
    W = output_proj['path'][0]['weight'].to(args.device)
    b = output_proj['path'][0]['bias'].to(args.device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in DataLoader(test_ds, batch_size=args.saga_batch_size):
            feats, labels = feats.to(args.device), labels.to(args.device)
            logits = feats @ W.T + b
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    test_acc = correct / total
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Save metrics
    metrics = {
        'val_loss': best_val_loss.item(),
        'test_acc': test_acc,
        'concept_sparsity': {
            'train': (concept_labels_train == 1).float().mean().item(),
            'val': (concept_labels_val == 1).float().mean().item(),
            'test': (concept_labels_test == 1).float().mean().item()
        },
        'training_metrics': output_proj['path'][0]['metrics']
    }
    with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save concept features and labels
    torch.save({
        'train_features': train_c.cpu(),
        'train_labels': train_targets,
        'val_features': val_c.cpu(),
        'val_labels': val_targets,
        'test_features': test_c.cpu(),
        'test_labels': test_targets
    }, os.path.join(save_dir, "concept_data.pt"))

if __name__ == "__main__":
    sys.stdout = LoggerWriter("INFO")
    sys.stderr = LoggerWriter("DEBUG")

    parser = argparse.ArgumentParser(description="Train CBM model")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--concept_set", type=str, default="./data/generate_concept/concept/cifar10_concepts_gpt-4o_final.json", help="path to concept set name")
    parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

    parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
    parser.add_argument("--cbl_batch_size", type=int, default=512, help="Batch size to use when learning concept bottleneck layer")

    parser.add_argument("--feature_layer", type=str, default='layer4', 
                        help="Which layer to collect activations from. Should be the name of second to last layer in the model")
    parser.add_argument(
        "--use_penultimate", 
        action="store_true",
        default=False,  
        help="Use penultimate layer (default: False)"
    )
    parser.add_argument(
        "--K_indep",
        type=int,
        default=5,
        help="Max number of independent components per class(default: 5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="adam weight_decay"
    )
    parser.add_argument("--activation_dir", type=str, default='/root/autodl-tmp/saved_activations8', help="save location for backbone and CLIP activations")
    parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
    parser.add_argument("--cbl_steps", type=int, default=20000, help="max steps to train the concept bottleneck layer for")
    parser.add_argument("--cbl_lr", type=float, default=0.001, help="cbl_lr")
    parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
    parser.add_argument("--n_iters", type=int, default=10000, help="How many iterations to run the final layer solver for")
    parser.add_argument("--Tconf", type=float, default=0.20, help="Threshold for filtering and labeling")
    parser.add_argument("--Tmerge", type=float, default=0.9998, help="Threshold for merging concepts")
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