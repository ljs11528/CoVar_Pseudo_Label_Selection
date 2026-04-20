import argparse
import os
import pytorch_lightning as pl 
import yaml
import torch
from dataset.semi import SemiDataset
from model.model_helper import ModelBuilder
from train.semi_supervised_train import SemiModule
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
from supervised import find_latest_checkpoint

def main():   
    parser = argparse.ArgumentParser(description='Separating Optimization for Reliable Prediction\
                                      in Semi-supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--labeled_id_path', type=str, required=True)
    parser.add_argument('--unlabeled_id_path', type=str, required=True)
    parser.add_argument('--val_id_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--enable_visual_artifacts', action='store_true', help='Enable pseudo-label diagnostics sampling and artifact export during training')
    parser.add_argument('--threshold_strategy', type=str, default='dynamic', choices=['dynamic', 'fixed'],
                        help='Pseudo-label threshold strategy: dynamic (max_confidence + residual variance) or fixed (max_confidence only)')
    parser.add_argument('--fixed_threshold', type=float, default=0.95,
                        help='Fixed confidence threshold used when --threshold_strategy fixed')
    parser.add_argument('--g_alpha', type=float, default=1.0,
                        help='Scale parameter for residual variance')
    parser.add_argument('--monitor_thresholds', type=str, default='0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99',
                        help='Comma-separated confidence thresholds for monitoring pseudo-label metrics during training')
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    monitor_thresholds = [float(x.strip()) for x in args.monitor_thresholds.split(',') if x.strip()]

    # 使用WandbLogger替换TensorBoardLogger
    logger = WandbLogger(project="CSL-CoVar", save_dir=args.save_path)
    logger.experiment.config.update(cfg)

    pl.seed_everything(42, workers=True)

    print(f"[pid {os.getpid()}] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    model = ModelBuilder(cfg['model'])
    
    trainset_u = SemiDataset(**{**cfg['dataset'], 'mode': 'train_u', 'id_path': args.unlabeled_id_path})
    trainset_l = SemiDataset(**{**cfg['dataset'], 'mode': 'train_l', 'id_path': args.labeled_id_path, 'nsample': len(trainset_u.ids)})
    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', args.val_id_path)

    # Reduce worker count to avoid /dev/shm bus errors inside small-shm containers
    num_workers = int(os.environ.get('NUM_WORKERS', '4'))
    pin_memory = os.environ.get('PIN_MEMORY', '1').lower() in ('1', 'true', 'yes')
    persistent_workers = num_workers > 0

    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    train_loaders = {
        'labeled': trainloader_l,
        'unlabeled': trainloader_u,
        'mixed': trainloader_u,
    }

    batch_iters = max(len(trainloader_l), len(trainloader_u)) // cfg['gpu_num']
    total_iters = batch_iters * cfg['epochs']
    train_module = SemiModule(**{
        **cfg['train'],
        'model': model,
        'save_path': args.save_path, 
        'batch_iters': batch_iters, 
        'total_iters':total_iters,
        'nclass': cfg['nclass'],
        'enable_visual_artifacts': args.enable_visual_artifacts,
        'threshold_strategy': args.threshold_strategy,
        'fixed_threshold': args.fixed_threshold,
        'g_alpha': args.g_alpha,
        'monitor_thresholds': monitor_thresholds,
        })
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],     
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false",
        benchmark=True,
        logger=logger,
        precision="bf16-mixed",
        sync_batchnorm=True,
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
        enable_checkpointing=True,
        log_every_n_steps= batch_iters // 32,
    )
    # 记录模型结构和超参数到wandb
    wandb.watch(model)

    checkpoint_path = find_latest_checkpoint(os.path.join(args.save_path, "checkpoints"))
    if trainer.is_global_zero and checkpoint_path != None:
        print("load checkpoint : ",checkpoint_path)

    trainer.fit(train_module, train_dataloaders=train_loaders, val_dataloaders=valloader, ckpt_path=checkpoint_path)

    # 训练结束后关闭wandb
    wandb.finish()

if __name__ == '__main__':
    main()