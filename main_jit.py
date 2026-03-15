import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # video frame generation
    parser.add_argument('--max_condition_frames', default=2, type=int,
                        help='Number of condition frames for video generation')
    parser.add_argument('--text_latent_dim', default=512, type=int,
                        help='Dimension of text latents (CLIP)')
    parser.add_argument('--use_text_condition', action='store_true',
                        help='Use text condition for CFG')
    parser.add_argument('--text_drop_prob', default=0.1, type=float,
                        help='Text latent dropout probability for CFG training')
    parser.add_argument('--condition_drop_prob', default=0.1, type=float,
        help='Condition frames dropout probability for CFG training')
    parser.add_argument('--use_condition_frames', action='store_true',
        help='Use condition frames instead of labels')
    parser.add_argument('--action_loss_weight', default=0.1, type=float,
        help='Weight for action prediction loss in multi-task learning')
    parser.add_argument('--action_self_condition_prob', default=0.0, type=float,
        help='Probability of using model-predicted frame as action condition during training (light self-forcing)')
    parser.add_argument('--freeze_backbone', action='store_true',
        help='Freeze all parameters except action modules (for fine-tuning)')
    parser.add_argument('--action_only_loss', action='store_true',
        help='Use only action loss (no image loss), all params trainable. For fine-tuning on action data.')
    parser.add_argument('--dataset_names', type=str, default='cup_arrangement_0,towel_folding_0,mouse_arrangement_0',
                        help='Comma-separated dataset names: UMI zarr names, PushT zarr name, or LIBERO hdf5 basenames (optional)')
    parser.add_argument('--dataset_type', type=str, default='umi', choices=['umi', 'pusht', 'libero10'],
                        help='Dataset type: umi, pusht, or libero10')
    parser.add_argument('--used_episode_indices_file', type=str, default='',
                        help='JSON file specifying which episodes to use (optional)')
    parser.add_argument('--action_dim', type=int, default=10,
                        help='Action dimension per step (default 10 for UMI, 2 for pushT)')
    parser.add_argument('--action_horizon', type=int, default=1,
                        help='Number of future action steps to predict (1=single-step, 8=multi-step)')
    parser.add_argument('--pusht_use_augmentation', action='store_true',
                        help='Enable PushT training augmentation (random crop + blur)')
    parser.add_argument('--pusht_random_crop_pad', type=int, default=4,
                        help='Padding size for PushT random crop augmentation')
    parser.add_argument('--pusht_blur_prob', type=float, default=0.2,
                        help='Probability of Gaussian blur in PushT augmentation')
    parser.add_argument('--pusht_blur_kernel_size', type=int, default=5,
                        help='Kernel size for PushT Gaussian blur augmentation (odd number)')
    parser.add_argument('--pusht_blur_sigma_min', type=float, default=0.1,
                        help='Minimum sigma for PushT Gaussian blur')
    parser.add_argument('--pusht_blur_sigma_max', type=float, default=1.5,
                        help='Maximum sigma for PushT Gaussian blur')
    parser.add_argument('--eval_pusht_success', action='store_true',
                        help='When action_dim=2, run pushT env evaluation for success rate (requires UVA)')
    parser.add_argument('--eval_libero_success', action='store_true',
                        help='Run LIBERO simulation evaluation during online eval (requires UVA+LIBERO)')
    parser.add_argument('--libero_n_train', type=int, default=1,
                        help='LIBERO eval: number of train episodes per task')
    parser.add_argument('--libero_n_train_vis', type=int, default=0,
                        help='LIBERO eval: number of train episodes with video')
    parser.add_argument('--libero_n_test', type=int, default=3,
                        help='LIBERO eval: number of test episodes per task')
    parser.add_argument('--libero_n_test_vis', type=int, default=0,
                        help='LIBERO eval: number of test episodes with video')
    parser.add_argument('--libero_max_steps', type=int, default=500,
                        help='LIBERO eval: max env steps')
    parser.add_argument('--libero_n_obs_steps', type=int, default=16,
                        help='LIBERO eval: observation horizon')
    parser.add_argument('--libero_n_action_steps', type=int, default=8,
                        help='LIBERO eval: action chunk size')
    parser.add_argument('--libero_test_start_seed', type=int, default=100000,
                        help='LIBERO eval: first test seed')
    parser.add_argument('--libero_fps', type=int, default=10,
                        help='LIBERO eval: rollout video FPS')
    parser.add_argument('--libero_use_augmentation', action='store_true',
                        help='Enable LIBERO training augmentation (random crop + blur)')
    parser.add_argument('--libero_random_crop_pad', type=int, default=4,
                        help='Padding size for LIBERO random crop augmentation')
    parser.add_argument('--libero_blur_prob', type=float, default=0.2,
                        help='Probability of Gaussian blur in LIBERO augmentation')
    parser.add_argument('--libero_blur_kernel_size', type=int, default=5,
                        help='Kernel size for LIBERO Gaussian blur augmentation (odd number)')
    parser.add_argument('--libero_blur_sigma_min', type=float, default=0.1,
                        help='Minimum sigma for LIBERO Gaussian blur')
    parser.add_argument('--libero_blur_sigma_max', type=float, default=1.5,
                        help='Maximum sigma for LIBERO Gaussian blur')
    parser.add_argument('--pusht_normalizer_path', type=str, default='',
                        help='Path to normalizer.pkl for pushT eval (optional)')
    parser.add_argument('--pusht_dataset_path', type=str, default='',
                        help='Path to pushT zarr for fitting normalizer (optional)')
    parser.add_argument('--pusht_wandb_video', action='store_true',
                        help='Log PushT rollout videos to wandb (like UVA)')
    parser.add_argument('--pusht_rollout_action_steps', type=int, default=8,
                        help='Number of actions executed per policy rollout chunk in PushT eval')

    # wandb
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging for training')
    parser.add_argument('--wandb_project', type=str, default='jit-opt',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: output_dir basename)')
    
    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--reset_epoch_on_resume', action='store_true',
                        help='When resuming, load model weights but reset start_epoch to 0')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank() #每个进程使用不同的种子，保证可复现性且防止各进程数据完全一致
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Init wandb (only on main process)
    use_wandb = getattr(args, 'wandb', False) and global_rank == 0
    if use_wandb:
        try:
            import wandb
            wandb_run_name = getattr(args, 'wandb_run_name', None) or os.path.basename(args.output_dir.rstrip('/'))
            wandb.init(
                project=getattr(args, 'wandb_project', 'jit-opt'),
                name=wandb_run_name,
                config=vars(args),
                mode="offline" if getattr(args, 'wandb_offline', False) else "online",
            )
            print(f"Wandb initialized: project={args.wandb_project}, run={wandb_run_name}")
        except ImportError:
            use_wandb = False
            print("wandb not installed, skipping wandb logging")
    args._use_wandb = use_wandb

    # Data loading: 支持两种模式
    if args.use_condition_frames:
        dataset_type = getattr(args, 'dataset_type', 'umi')
        if dataset_type == 'pusht':
            from dataset.pusht_video_dataset import PushTVideoDataset
            # PushT: data_path 为 pusht 目录，dataset_names 为 zarr 名（如 pusht_cchi_v7_replay）
            names = [n.strip() for n in args.dataset_names.split(',') if n.strip()]
            pusht_name = names[0] if names else 'pusht_cchi_v7_replay'
            pusht_path = os.path.join(args.data_path, pusht_name)
            if not pusht_path.endswith('.zarr'):
                pusht_path = pusht_path + '.zarr'
            if not os.path.exists(pusht_path) and os.path.exists(os.path.join(args.data_path, pusht_name)):
                pusht_path = os.path.join(args.data_path, pusht_name)
            dataset_train = PushTVideoDataset(
                dataset_path=pusht_path,
                max_condition_frames=args.max_condition_frames,
                image_size=args.img_size,
                split='train',
                action_horizon=getattr(args, 'action_horizon', 1),
                use_augmentation=getattr(args, 'pusht_use_augmentation', False),
                random_crop_pad=getattr(args, 'pusht_random_crop_pad', 4),
                blur_prob=getattr(args, 'pusht_blur_prob', 0.2),
                blur_kernel_size=getattr(args, 'pusht_blur_kernel_size', 5),
                blur_sigma_min=getattr(args, 'pusht_blur_sigma_min', 0.1),
                blur_sigma_max=getattr(args, 'pusht_blur_sigma_max', 1.5),
            )
        elif dataset_type == 'libero10':
            from dataset.libero10_video_dataset import Libero10VideoDataset
            dataset_names = [name.strip() for name in args.dataset_names.split(',') if name.strip()]
            dataset_train = Libero10VideoDataset(
                dataset_root_dir=args.data_path,
                max_condition_frames=args.max_condition_frames,
                image_size=args.img_size,
                split='train',
                action_horizon=getattr(args, 'action_horizon', 1),
                dataset_names=dataset_names if len(dataset_names) > 0 else None,
                use_augmentation=getattr(args, 'libero_use_augmentation', False),
                random_crop_pad=getattr(args, 'libero_random_crop_pad', 4),
                blur_prob=getattr(args, 'libero_blur_prob', 0.2),
                blur_kernel_size=getattr(args, 'libero_blur_kernel_size', 5),
                blur_sigma_min=getattr(args, 'libero_blur_sigma_min', 0.1),
                blur_sigma_max=getattr(args, 'libero_blur_sigma_max', 1.5),
            )
        else:
            from dataset.umi_video_dataset import UmiVideoDataset
            dataset_names = [name.strip() for name in args.dataset_names.split(',') if name.strip()]
            dataset_train = UmiVideoDataset(
                dataset_root_dir=args.data_path,
                max_condition_frames=args.max_condition_frames,
                image_size=args.img_size,
                split='train',
                dataset_names=dataset_names,
                used_episode_indices_file=args.used_episode_indices_file if args.used_episode_indices_file else None
            )
        print(f"Video dataset ({dataset_type}): {len(dataset_train)} samples")
        # Store action normalization stats for eval denormalization
        args._action_stats = getattr(dataset_train, 'action_stats', None)
    else:
        # ImageNet模式（兼容性）
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ])

        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    # Create validation dataloader for evaluation (only for video frame generation)
    data_loader_val = None
    if args.use_condition_frames and args.online_eval:
        dataset_type = getattr(args, 'dataset_type', 'umi')
        if dataset_type == 'pusht':
            from dataset.pusht_video_dataset import PushTVideoDataset
            names = [n.strip() for n in args.dataset_names.split(',') if n.strip()]
            pusht_name = names[0] if names else 'pusht_cchi_v7_replay'
            pusht_path = os.path.join(args.data_path, pusht_name)
            if not pusht_path.endswith('.zarr'):
                pusht_path = pusht_path + '.zarr'
            dataset_val = PushTVideoDataset(
                dataset_path=pusht_path,
                max_condition_frames=args.max_condition_frames,
                image_size=args.img_size,
                split='val',
                action_horizon=getattr(args, 'action_horizon', 1),
            )
        elif dataset_type == 'libero10':
            from dataset.libero10_video_dataset import Libero10VideoDataset
            dataset_names = [name.strip() for name in args.dataset_names.split(',') if name.strip()]
            dataset_val = Libero10VideoDataset(
                dataset_root_dir=args.data_path,
                max_condition_frames=args.max_condition_frames,
                image_size=args.img_size,
                split='val',
                action_horizon=getattr(args, 'action_horizon', 1),
                dataset_names=dataset_names if len(dataset_names) > 0 else None,
            )
        else:
            from dataset.umi_video_dataset import UmiVideoDataset
            dataset_names = [name.strip() for name in args.dataset_names.split(',') if name.strip()]
            dataset_val = UmiVideoDataset(
                dataset_root_dir=args.data_path,
                max_condition_frames=args.max_condition_frames,
                image_size=args.img_size,
                split='val',
                dataset_names=dataset_names,
                used_episode_indices_file=args.used_episode_indices_file if args.used_episode_indices_file else None
            )
        print(f"Validation dataset: {len(dataset_val)} samples")
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    # Pass freeze_backbone flag to Denoiser (will be set later, but prepare args)
    model = Denoiser(args)
    
    # Set freeze_backbone in denoiser after creation (if needed)
    if args.freeze_backbone:
        model.freeze_backbone = True

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256  #更大的批次通常需要更大的学习率

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    # Wrap model in DDP only if using distributed training
    if args.distributed:
        # 启用find_unused_parameters以处理条件性使用的参数（如text_embedder）
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True  # 某些参数可能在某些batch中未使用（如text_embedder）
        )
        model_without_ddp = model.module  # DDP 包装后，原始模型被存储在 model.module 中
    else:
        model_without_ddp = model  # 非分布式模式下直接使用模型

    # Resume from checkpoint if provided
    # Support both file path and directory path
    checkpoint_path = None
    checkpoint_optimizer = None
    if args.resume:
        if os.path.isfile(args.resume):
            # Direct file path
            checkpoint_path = args.resume
        elif os.path.isdir(args.resume):
            # Directory path, look for checkpoint-last.pth
            checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Use strict=False to allow loading checkpoints without action_head (for backward compatibility)
        # or with different action_dim (e.g., loading UMI checkpoint for pushT training)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if missing_keys:
            print(f"Note: Missing keys (newly added parameters): {missing_keys[:5]}...")
            if any('action_head' in key for key in missing_keys):
                print(f"Note: action_head will be reinitialized (action_dim={getattr(args, 'action_dim', 10)}, action_horizon={getattr(args, 'action_horizon', 1)})")
        if unexpected_keys:
            print(f"Note: Unexpected keys (ignored): {unexpected_keys[:5]}...")
            if any('action_head' in key for key in unexpected_keys):
                print(f"Note: Old action_head ignored (shape mismatch, new action_dim={getattr(args, 'action_dim', 10)}, action_horizon={getattr(args, 'action_horizon', 1)})")

        # Load EMA parameters (only for existing parameters)
        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        # Initialize EMA for all parameters (existing ones from checkpoint, new ones from current model)
        ema_params1_list = []
        ema_params2_list = []
        for name, param in model_without_ddp.named_parameters():
            if name in ema_state_dict1:
                ema_params1_list.append(ema_state_dict1[name].cuda())
            else:
                # New parameter (e.g., action_head), initialize EMA with current value
                ema_params1_list.append(param.data.clone().cuda())
            if name in ema_state_dict2:
                ema_params2_list.append(ema_state_dict2[name].cuda())
            else:
                ema_params2_list.append(param.data.clone().cuda())
        model_without_ddp.ema_params1 = ema_params1_list
        model_without_ddp.ema_params2 = ema_params2_list
        print("Resumed checkpoint from", checkpoint_path)
        
        # Store checkpoint info for later optimizer loading
        checkpoint_epoch = checkpoint.get('epoch', 0)
        checkpoint_optimizer = checkpoint.get('optimizer', None)
        # Reset epoch counter for fine-tuning or explicit reset request.
        if (getattr(args, 'action_only_loss', False)
                or getattr(args, 'freeze_backbone', False)
                or getattr(args, 'reset_epoch_on_resume', False)):
            args.start_epoch = 0
            print(f"Fine-tuning mode: resetting start_epoch to 0 (checkpoint was at epoch {checkpoint_epoch})")
        else:
            args.start_epoch = checkpoint_epoch + 1
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        args.start_epoch = 0
        print("Training from scratch")
    
    # Freeze backbone if requested (only train action modules)
    if args.freeze_backbone:
        print("="*60)
        print("Freezing backbone parameters, only training action modules")
        print("="*60)
        frozen_params = 0
        trainable_params = 0
        for name, param in model_without_ddp.named_parameters():
            if ('action_head' in name
                or 'action_pooler' in name
                or 'action_token_proj' in name
                or 'action_token_posemb' in name
                or 'action_t_embedder' in name):
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"Frozen parameters: {frozen_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print("="*60)
        
        # Set freeze_backbone flag in denoiser for loss computation optimization
        # This will be passed to Denoiser during initialization, but we also set it here
        # for models that are already created
        if hasattr(model_without_ddp, 'freeze_backbone'):
            model_without_ddp.freeze_backbone = True
    
    # Set up optimizer with weight decay adjustment for bias and norm layers
    if args.freeze_backbone:
        # Only optimize trainable parameters (action modules)
        trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
        trainable_names = [name for name, p in model_without_ddp.named_parameters() if p.requires_grad]
        
        # Manually create param groups with weight decay
        decay = []
        no_decay = []
        for name, param in zip(trainable_names, trainable_params):
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        
        param_groups = [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': args.weight_decay}
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print("Optimizer created with only trainable parameters (action modules)")
    else:
        param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    # Try to load optimizer state from checkpoint (if available and structure matches)
    if checkpoint_path and os.path.exists(checkpoint_path) and checkpoint_optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint_optimizer)
            print("Loaded optimizer state from checkpoint!")
        except Exception as e:
            print(f"Warning: Could not load optimizer state (structure may differ): {e}")
            print("Continuing with fresh optimizer state (this is normal if resuming from different training stage)")

    # Evaluate generation
    if args.evaluate_gen:  #是用这一个参数区分出eval和train的，因为都写在main里
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch) #确保每个 epoch 的数据顺序不同

        #训练
        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs: #最新模型
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0: #定期保存
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, data_loader_val=data_loader_val)
            torch.cuda.empty_cache()

            # PushT success rate evaluation (action_dim=2 only).
            # DDP note: PushT env rollout can exceed NCCL collective timeout.
            # Use a file-based rendezvous to keep ranks in sync without collectives.
            if getattr(args, 'eval_pusht_success', False) and getattr(args, 'action_dim', 10) == 2:
                sync_file = os.path.join(args.output_dir, f".pusht_eval_done_epoch_{epoch}.sync")
                if misc.is_main_process():
                    if os.path.exists(sync_file):
                        os.remove(sync_file)
                    try:
                        from engine_jit import run_pusht_success_eval
                        run_pusht_success_eval(model_without_ddp, args, epoch, log_writer)
                    except Exception as e:
                        print(f"PushT success eval skipped: {e}")
                    finally:
                        try:
                            with open(sync_file, "w", encoding="utf-8") as f:
                                f.write("done\n")
                        except Exception:
                            pass
                elif torch.distributed.is_initialized():
                    # Wait for rank0 to finish PushT eval without triggering NCCL timeout.
                    wait_s = 0
                    while not os.path.exists(sync_file):
                        time.sleep(2)
                        wait_s += 2
                        if wait_s % 120 == 0:
                            print(f"[rank {misc.get_rank()}] waiting PushT eval sync ({wait_s}s)")

            if getattr(args, 'eval_libero_success', False) and getattr(args, 'dataset_type', 'umi') == 'libero10':
                sync_file = os.path.join(args.output_dir, f".libero_eval_done_epoch_{epoch}.sync")
                if misc.is_main_process():
                    if os.path.exists(sync_file):
                        os.remove(sync_file)
                    try:
                        from engine_jit import run_libero_success_eval
                        run_libero_success_eval(model_without_ddp, args, epoch, log_writer)
                    except Exception as e:
                        print(f"LIBERO eval skipped: {e}")
                    finally:
                        try:
                            with open(sync_file, "w", encoding="utf-8") as f:
                                f.write("done\n")
                        except Exception:
                            pass
                elif torch.distributed.is_initialized():
                    wait_s = 0
                    while not os.path.exists(sync_file):
                        time.sleep(2)
                        wait_s += 2
                        if wait_s % 120 == 0:
                            print(f"[rank {misc.get_rank()}] waiting LIBERO eval sync ({wait_s}s)")

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)