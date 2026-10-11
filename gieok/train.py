"""Training script for GIEOK."""

import argparse
import logging
import torch
import torch.nn as nn
import yaml
from pathlib import Path

from gieok.model import GIEOK
from gieok.data import load_manifest, load_cues

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        speech_features = batch['speech'].to(device)
        cue_embeddings = batch['cues'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(speech_features, cue_embeddings)

        # Classification loss
        logits = outputs['logits']
        loss = nn.CrossEntropyLoss()(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['training']['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

    return total_loss / (batch_idx + 1)


def evaluate(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            speech_features = batch['speech'].to(device)
            cue_embeddings = batch['cues'].to(device)
            labels = batch['label']

            outputs = model(speech_features, cue_embeddings)
            preds = outputs['logits'].argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    from gieok.metrics import compute_metrics
    metrics = compute_metrics(all_preds, all_targets)
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GIEOK model")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--cues", type=str, required=True, help="Path to cues CSV")
    parser.add_argument("--config", type=str, default="gieok/configs/gieok.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Override epochs")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config['training']['epochs'] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading manifest from {args.manifest}")
    manifest = load_manifest(args.manifest)

    logger.info(f"Loading cues from {args.cues}")
    cues = load_cues(args.cues)

    # Initialize model
    model = GIEOK(**config['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info(f"Starting training for {config['training']['epochs']} epochs")

    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        # Placeholder: actual training would load batches from manifest
        logger.info("  (Placeholder training)")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            model.save(str(checkpoint_path), config['model'])
            logger.info(f"  Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
