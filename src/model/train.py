"""T5-small fine-tuning for generative cover-song retrieval.

  - Extends T5's vocab with N_LEVELS * CODEBOOK_SIZE SemID tokens
  - Fine-tunes on (query_semid -> target_semid) ordered pairs from the train CSV
  - Validates on val CSV, early-stops on val loss

Run one invocation per SemID condition (random / mert / encodec). The
condition is implicit in the CSV paths + codebook size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    set_seed,
)

from src.model.dataset import (
    SemIDPairsDataset,
    build_semid_vocab,
    extend_tokenizer_with_semids,
)


def _reinit_semid_embeddings(model, tokenizer, n_levels, codebook_size, seed):
    """Re-initialize SemID-token embedding rows.

    HF resize_token_embeddings defaults to MEAN resizing: every new token starts
    at the mean of the original embeddings, i.e. all SemID tokens are identical.
    That starves the encoder of input signal and causes mode collapse. We instead
    sample each SemID embedding from the per-dimension Gaussian of the original
    vocabulary, so they start differentiated and at the right magnitude.
    """
    sem_tokens = build_semid_vocab(n_levels, codebook_size)
    sem_ids = [tokenizer.convert_tokens_to_ids(t) for t in sem_tokens]
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight  # tied: input == lm_head
        mask = torch.ones(emb.shape[0], dtype=torch.bool)
        for i in sem_ids:
            mask[i] = False
        orig = emb[mask]
        mu = orig.mean(dim=0)
        sigma = orig.std(dim=0)
        noise = torch.randn(len(sem_ids), emb.shape[1], generator=g)
        emb[sem_ids] = mu + sigma * noise
    print(f"  re-initialized {len(sem_ids)} SemID embeddings "
          f"(orig std/dim mean={sigma.mean():.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-model", default="t5-small")
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256,
                        help="Use 256 for random/mert, 1024 for encodec.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="3e-4 validated by local smoke test: with the SemID embedding "
                             "re-init fix, loss breaks the 3.25 collapse floor cleanly at this LR.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Bumped from 200: extended vocab benefits from a longer warmup.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Kept at 0: generative retrieval is a memorization task "
                             "(the model IS the index), so confident targets are desired. "
                             "The real mode-collapse fix is SemID embedding re-init.")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                        help="auto picks bf16 on Ampere+ CUDA (T5 is known to NaN in fp16). "
                             "Use fp32 if you hit instability.")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="If >0, cap training at this many steps (smoke test).")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base tokenizer/model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    n_added = extend_tokenizer_with_semids(tokenizer, args.n_levels, args.codebook_size)
    print(f"  added {n_added} SemID tokens; total vocab = {len(tokenizer)}")

    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    _reinit_semid_embeddings(model, tokenizer, args.n_levels, args.codebook_size, args.seed)

    print(f"Loading train: {args.train_csv}")
    train_ds = SemIDPairsDataset(args.train_csv, tokenizer)
    print(f"  {len(train_ds)} training pairs")
    print(f"Loading val:   {args.val_csv}")
    val_ds = SemIDPairsDataset(args.val_csv, tokenizer)
    print(f"  {len(val_ds)} val pairs")

    # Choose precision: T5 is known to NaN in fp16 (activation underflow). Use
    # bf16 on Ampere+ CUDA; fall back to fp32 elsewhere.
    use_fp16 = False
    use_bf16 = False
    if args.precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_bf16 = True
    elif args.precision == "fp16":
        use_fp16 = True
    elif args.precision == "bf16":
        use_bf16 = True
    # fp32 leaves both False
    print(f"Precision: fp16={use_fp16}, bf16={use_bf16}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print(f"Training. fp16={use_fp16}, device={trainer.args.device}")
    train_result = trainer.train()
    print(f"Training done. Final train loss: {train_result.training_loss:.4f}")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save a small JSON of the run config for later (inference needs n_levels + codebook_size)
    cfg = {
        "base_model": args.base_model,
        "n_levels": args.n_levels,
        "codebook_size": args.codebook_size,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "vocab_size": len(tokenizer),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved model + tokenizer + run_config.json to {out_dir}")


if __name__ == "__main__":
    main()
