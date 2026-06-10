#!/usr/bin/env python3
"""
Fine-tune a cross-encoder reranker on PF2E rules triplets.

Usage:
    uv run python train_reranker.py \\
        --model BAAI/bge-reranker-v2-m3 \\
        --data ../training_data/dataset.jsonl \\
        --output ./reranker-finetuned \\
        --push-to-hub kaylebor/pf2e-codex-reranker
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from huggingface_hub import HfApi


class RerankerTripletDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append((obj["query"], obj["pos"], obj["neg"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, pos, neg = self.samples[idx]
        return query, pos, neg


def collate_fn(batch, tokenizer, max_length):
    queries, pos_texts, neg_texts = zip(*batch)

    # Batch pos and neg together in ONE forward call (ROCm fix: separate
    # forward passes can corrupt LayerNorm gradients)
    all_queries = list(queries) + list(queries)
    all_texts = list(pos_texts) + list(neg_texts)
    enc = tokenizer(
        all_queries, all_texts,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "batch_size": len(queries),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
    )
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model.to(device)

    # Create dataset
    dataset = RerankerTripletDataset(args.data, tokenizer, args.max_length)
    print(f"Loaded {len(dataset)} training samples")

    # Split train/val
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(int(len(indices) * args.val_split), 1)
    val_indices = set(indices[:split])
    train_indices = indices[split:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, list(val_indices))

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=0,
    )
    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = batch["batch_size"]

            # Single forward pass — batch pos and neg together (ROCm LayerNorm fix)
            all_scores = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            pos_score = all_scores[:batch_size]
            neg_score = all_scores[batch_size:]

            # Margin ranking loss: pos_score should be > neg_score by margin
            loss = torch.clamp(args.margin - pos_score + neg_score, min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_size = batch["batch_size"]

                with torch.no_grad():
                    all_scores = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                    pos_score = all_scores[:batch_size]
                    neg_score = all_scores[batch_size:]

                loss = torch.clamp(args.margin - pos_score + neg_score, min=0).mean()
                total_val_loss += loss.item()
                correct += (pos_score > neg_score).sum().item()
                total += pos_score.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total * 100 if total > 0 else 0
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.1f}%")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"Saved best model to {output_dir} (val_loss={best_val_loss:.4f})")

    # Push to Hub
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        model = AutoModelForSequenceClassification.from_pretrained(args.output)
        tokenizer = AutoTokenizer.from_pretrained(args.output)
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3", help="Base model name")
    parser.add_argument("--data", default="../training_data/dataset.jsonl", help="Training data JSONL")
    parser.add_argument("--output", default="./reranker-finetuned", help="Output directory")
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo to push to")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep small for 24GB VRAM)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for ranking loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
