import os
import argparse
import torch

# fbgemm_gpu must be imported first to register custom ops used by Research HSTU
try:
    import fbgemm_gpu
    torch.ops.fbgemm.asynchronous_complete_cumsum
    print("[✓] fbgemm_gpu loaded successfully")
except (ImportError, AttributeError) as e:
    print(f"[WARNING] fbgemm_gpu not available: {e}")
    print("Install with: pip install fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cu124")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("[Note] DeepSpeed not available - will use regular PyTorch training")

from src.config import UniGCRConfig
from src.data_amazon import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed, setup_distributed, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="UniGCR Training")
    parser.add_argument('--data_path', type=str, default='data/train_sequences.json')
    parser.add_argument('--grid_mapping', type=str, default='data/semantic_id_kmean.pt')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, evaluate a saved checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')

    if DEEPSPEED_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)
    else:
        parser.add_argument('--deepspeed_config', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    setup_distributed()

    conf = UniGCRConfig()
    conf.data_path = args.data_path
    conf.grid_mapping_path = args.grid_mapping
    set_seed(conf.seed)

    if is_main_process():
        print(f"Data:         {conf.data_path}")
        print(f"GRID mapping: {conf.grid_mapping_path}")

    train_dl, val_dl = get_dataloaders(conf)

    if is_main_process():
        print(f"Semantic vocab size: {conf.sem_total_vocab}")

    model = UniGCRModel(conf)
    trainer = UniGCRTrainer(config=conf, args=args, model=model,
                            train_loader=train_dl, val_loader=val_dl)

    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"embed_dim   : {conf.embed_dim}")
        print(f"hstu_layers : {conf.hstu_layers}  heads: {conf.hstu_heads}")
        print(f"max_seq_len : {conf.max_seq_len}  ({(conf.max_seq_len-1)//conf.sem_id_layers} items)")
        print(f"batch_size  : {conf.batch_size}")
        print(f"parameters  : {num_params:,}")

    if args.eval_only:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return

        if is_main_process():
            print(f"\n[Eval-Only] Loading {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        eval_results = trainer.evaluate(topk=10)

        if is_main_process():
            print("=" * 40)
            print(f"GR Loss : {eval_results['val_gr_loss']:.4f}")
            print(f"Hit@10  : {eval_results['Hit@10']:.4f}")
            print(f"NDCG@10 : {eval_results['NDCG@10']:.4f}")
            if conf.enable_ctr:
                print(f"AUC     : {eval_results.get('AUC', 0):.4f}")
                print(f"LogLoss : {eval_results.get('LogLoss', 0):.4f}")
            print("=" * 40)
    else:
        if is_main_process():
            print("Starting training...")
        trainer.train()


if __name__ == "__main__":
    main()
