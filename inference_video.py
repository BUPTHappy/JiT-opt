import os
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
from dataset.umi_video_dataset import UmiVideoDataset
from denoiser import Denoiser
from model_jit import JiT_models
import copy
import shutil

try:
    import torch_fidelity
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False
    print("Warning: torch_fidelity not available. FID metrics will be skipped.")


def get_args_parser():
    parser = argparse.ArgumentParser('JiT Video Frame Generation Inference', add_help=False)
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., checkpoint-last.pth)')
    parser.add_argument('--model', type=str, default='JiT-B/16',
                        choices=['JiT-B/16', 'JiT-B/32', 'JiT-L/16', 'JiT-L/32', 'JiT-XL/16', 'JiT-XL/32'],
                        help='Model architecture')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--max_condition_frames', type=int, default=2,
                        help='Number of condition frames')
    
    parser.add_argument('--data_path', type=str, default='/workspace/data/umi',
                        help='Path to UMI dataset')
    parser.add_argument('--dataset_names', type=str, default='cup_arrangement_0,towel_folding_0,mouse_arrangement_0',
                        help='Comma-separated dataset names')
    parser.add_argument('--image_key', type=str, default=None,
                        help='Image key in zarr data group (default: auto-detect)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split to use')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Output directory for generated frames')
    
    parser.add_argument('--sampling_method', type=str, default='heun',
                        choices=['euler', 'heun'],
                        help='Sampling method')
    parser.add_argument('--num_sampling_steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--cfg', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--noise_scale', type=float, default=1.0,
                        help='Noise scale for initialization')
    parser.add_argument('--t_eps', type=float, default=5e-2,
                        help='Minimum timestep')
    parser.add_argument('--interval_min', type=float, default=0.1,
                        help='Minimum interval for CFG')
    parser.add_argument('--interval_max', type=float, default=1.0,
                        help='Maximum interval for CFG')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model for inference')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute FID and other metrics between generated and target images')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save individual sample images (set to False to save disk space)')
    
    # LIBERO support
    parser.add_argument('--dataset_type', type=str, default='umi', choices=['umi', 'libero'],
                        help='Dataset type: umi (zarr, for UMI/PushT) or libero (HDF5)')
    parser.add_argument('--use_text_condition', action='store_true',
                        help='Use text condition (for LIBERO language-conditioned generation)')
    parser.add_argument('--text_latent_dim', type=int, default=512,
                        help='Text latent dimension (CLIP)')

    return parser


def load_checkpoint(checkpoint_path, model, device, use_ema=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if use_ema and 'model_ema1' in checkpoint:
        ema_state_dict1 = checkpoint['model_ema1']
        # Remove 'net.' prefix from keys if present
        if any(k.startswith('net.') for k in ema_state_dict1.keys()):
            ema_state_dict1 = {k[4:] if k.startswith('net.') else k: v for k, v in ema_state_dict1.items()}
        missing_keys, unexpected_keys = model.net.load_state_dict(ema_state_dict1, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading EMA: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading EMA: {unexpected_keys[:5]}...")
        print("Loaded EMA model (ema1)")
    else:
        model_state_dict = checkpoint['model']
        # Remove 'net.' prefix from keys if present
        if any(k.startswith('net.') for k in model_state_dict.keys()):
            model_state_dict = {k[4:] if k.startswith('net.') else k: v for k, v in model_state_dict.items()}
        missing_keys, unexpected_keys = model.net.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading model: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading model: {unexpected_keys[:5]}...")
        print("Loaded regular model")
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    return model


def generate_video_frames(model, dataloader, args):
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create folders for generated and target images (for FID calculation)
    generated_folder = os.path.join(args.output_dir, "generated")
    target_folder = os.path.join(args.output_dir, "target")
    if args.compute_metrics:
        os.makedirs(generated_folder, exist_ok=True)
        os.makedirs(target_folder, exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= args.num_samples:
                break
            
            condition_frames = batch['condition_frames'].to(args.device)
            target_frame = batch['target_frame'].to(args.device)
            
            # Ensure condition_frames are in [-1, 1] range (dataset already normalizes, but double-check)
            # condition_frames should be (N, max_condition_frames, C, H, W)
            if condition_frames.max() > 1.1 or condition_frames.min() < -1.1:
                print(f"Warning: condition_frames not in [-1,1] range: min={condition_frames.min():.3f}, max={condition_frames.max():.3f}")
                condition_frames = torch.clamp(condition_frames, -1.0, 1.0)
            
            batch_size = condition_frames.shape[0]
            actual_batch_size = min(batch_size, args.num_samples - sample_count)
            
            condition_frames = condition_frames[:actual_batch_size]
            target_frame = target_frame[:actual_batch_size]

            # Handle text latents (for LIBERO language conditioning)
            text_latents = None
            if 'text_latents' in batch:
                text_latents = batch['text_latents'].to(args.device)[:actual_batch_size]
            
            print(f"Generating batch {batch_idx + 1}, samples {sample_count + 1}-{sample_count + actual_batch_size}")
            print(f"  condition_frames shape: {condition_frames.shape}, range: [{condition_frames.min():.3f}, {condition_frames.max():.3f}]")
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generated_frames = model.generate(condition_frames=condition_frames, text_latents=text_latents)
            
            print(f"  generated_frames shape: {generated_frames.shape}, range: [{generated_frames.min():.3f}, {generated_frames.max():.3f}]")
            
            # Clamp to [-1, 1] first (model output should be in this range)
            generated_frames = torch.clamp(generated_frames, -1.0, 1.0)
            # Then normalize to [0, 1] for image saving
            generated_frames = (generated_frames + 1) / 2
            generated_frames = torch.clamp(generated_frames, 0, 1)
            generated_frames = generated_frames.detach().cpu()
            
            print(f"  After normalization: range: [{generated_frames.min():.3f}, {generated_frames.max():.3f}]")
            
            condition_frames_cpu = (condition_frames + 1) / 2
            condition_frames_cpu = torch.clamp(condition_frames_cpu, 0, 1).detach().cpu()
            target_frame_cpu = (target_frame + 1) / 2
            target_frame_cpu = torch.clamp(target_frame_cpu, 0, 1).detach().cpu()
            
            for i in range(actual_batch_size):
                sample_id = sample_count + i
                
                # Save for FID calculation
                if args.compute_metrics:
                    gen_img = generated_frames[i].numpy().transpose(1, 2, 0)
                    gen_img = (gen_img * 255).astype(np.uint8)
                    Image.fromarray(gen_img).save(
                        os.path.join(generated_folder, f'{sample_id:05d}.png')
                    )
                    
                    target_img = target_frame_cpu[i].numpy().transpose(1, 2, 0)
                    target_img = (target_img * 255).astype(np.uint8)
                    Image.fromarray(target_img).save(
                        os.path.join(target_folder, f'{sample_id:05d}.png')
                    )
                
                # Save individual samples (if enabled)
                if args.save_images and sample_id < 20:  # Only save first 20 for visualization
                    sample_dir = os.path.join(args.output_dir, f'sample_{sample_id:05d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    for j in range(args.max_condition_frames):
                        cond_img = condition_frames_cpu[i, j].numpy().transpose(1, 2, 0)
                        cond_img = (cond_img * 255).astype(np.uint8)
                        Image.fromarray(cond_img).save(
                            os.path.join(sample_dir, f'condition_frame_{j:02d}.png')
                        )
                    
                    gen_img = generated_frames[i].numpy().transpose(1, 2, 0)
                    gen_img = (gen_img * 255).astype(np.uint8)
                    Image.fromarray(gen_img).save(
                        os.path.join(sample_dir, 'generated_frame.png')
                    )
                    
                    target_img = target_frame_cpu[i].numpy().transpose(1, 2, 0)
                    target_img = (target_img * 255).astype(np.uint8)
                    Image.fromarray(target_img).save(
                        os.path.join(sample_dir, 'target_frame.png')
                    )
            
            sample_count += actual_batch_size
    
    print(f"Generated {sample_count} samples in {args.output_dir}")
    
    # Compute metrics if requested
    if args.compute_metrics and HAS_TORCH_FIDELITY:
        print("\n" + "="*60)
        print("Computing evaluation metrics...")
        print("="*60)
        try:
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=generated_folder,
                input2=target_folder,
                cuda=torch.cuda.is_available(),
                fid=True,
                isc=True,
                kid=False,
                prc=False,
                verbose=True,
            )
            
            fid = metrics_dict.get('frechet_inception_distance', None)
            is_score = metrics_dict.get('inception_score_mean', None)
            
            print("\n" + "="*60)
            print("Evaluation Results:")
            print("="*60)
            if fid is not None:
                print(f"FID (Frechet Inception Distance): {fid:.4f}")
                print(f"  Lower is better (0 = identical distributions)")
            if is_score is not None:
                print(f"IS (Inception Score): {is_score:.4f}")
                print(f"  Higher is better")
            print("="*60)
            
            # Save metrics to file
            metrics_file = os.path.join(args.output_dir, "metrics.txt")
            with open(metrics_file, 'w') as f:
                f.write("Evaluation Metrics\n")
                f.write("="*60 + "\n")
                if fid is not None:
                    f.write(f"FID: {fid:.4f}\n")
                if is_score is not None:
                    f.write(f"IS: {is_score:.4f}\n")
                f.write(f"Number of samples: {sample_count}\n")
            print(f"\nMetrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
    elif args.compute_metrics and not HAS_TORCH_FIDELITY:
        print("Warning: torch_fidelity not available. Install it to compute FID metrics.")
        print("  pip install git+https://github.com/LTH14/torch-fidelity.git")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    print(f"Using device: {device}")

    # Create dataset based on type
    dataset_type = getattr(args, 'dataset_type', 'umi')
    use_text_condition = getattr(args, 'use_text_condition', False)
    text_latent_dim = getattr(args, 'text_latent_dim', 512)

    if dataset_type == 'libero':
        from dataset.libero_video_dataset import LiberoVideoDataset
        dataset = LiberoVideoDataset(
            dataset_path=args.data_path,
            max_condition_frames=args.max_condition_frames,
            image_size=args.img_size,
            split=args.split,
            use_text_condition=use_text_condition,
            text_latent_dim=text_latent_dim,
        )
    else:
        dataset_names = args.dataset_names.split(',')
        dataset = UmiVideoDataset(
            dataset_root_dir=args.data_path,
            max_condition_frames=args.max_condition_frames,
            image_size=args.img_size,
            split=args.split,
            dataset_names=dataset_names,
            image_key=getattr(args, 'image_key', None),
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    model_args = argparse.Namespace(
        model=args.model,
        img_size=args.img_size,
        max_condition_frames=args.max_condition_frames,
        text_latent_dim=text_latent_dim,
        use_text_condition=use_text_condition,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        condition_drop_prob=0.0,
        label_drop_prob=0.0,
        text_drop_prob=0.0,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
        P_mean=-0.8,
        P_std=0.8,
        ema_decay1=0.9999,
        ema_decay2=0.9996
    )
    
    model = Denoiser(model_args)
    model = model.to(device)
    
    model = load_checkpoint(args.checkpoint, model, device, use_ema=args.use_ema)
    
    generate_video_frames(model, dataloader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('JiT Video Frame Generation Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
