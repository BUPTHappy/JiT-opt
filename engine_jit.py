import math
import sys
import os
import shutil

import torch
import numpy as np
from PIL import Image

import util.misc as misc
import util.lr_sched as lr_sched
try:
    import torch_fidelity
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False
    print("Warning: torch_fidelity not available. FID/IS metrics will be skipped.")
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # 处理数据：支持两种模式兼容
        if isinstance(batch, dict):
            # 视频帧模式
            condition_frames = batch['condition_frames'].to(device, non_blocking=True)
            target_frame = batch['target_frame'].to(device, non_blocking=True)
            
            # Dataset already normalizes to [-1, 1], just ensure float32
            condition_frames = condition_frames.to(torch.float32)
            target_frame = target_frame.to(torch.float32)
            
            text_latents = None
            if 'text_latents' in batch and batch['text_latents'] is not None:
                text_latents = batch['text_latents'].to(device, non_blocking=True)
            
            # 获取action数据（如果存在）
            action_gt = None
            if 'action' in batch and batch['action'] is not None:
                action_gt = batch['action'].to(device, non_blocking=True)
            
            labels = None
        else:
            # ImageNet模式（兼容性）
            x, labels = batch
            x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
            x = x * 2.0 - 1.0
            labels = labels.to(device, non_blocking=True)
            condition_frames = None
            target_frame = x
            text_latents = None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(target_frame, labels=labels, condition_frames=condition_frames, 
                        text_latents=text_latents, action_gt=action_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        # 诊断信息：每1000步打印一次详细的loss统计
        if data_iter_step % 1000 == 0 and data_iter_step > 0:
            # 计算loss的统计信息
            with torch.no_grad():
                # 重新计算loss的统计（这里简化，实际可以从metric_logger获取）
                print(f"[Diagnostic] Step {data_iter_step}: loss={loss_value:.8f}, "
                      f"global_avg={metric_logger.meters['loss'].global_avg:.8f}, "
                      f"median={metric_logger.meters['loss'].median:.8f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                if getattr(args, '_use_wandb', False):
                    try:
                        import wandb
                        wandb.log({'train_loss': loss_value_reduce, 'lr': lr}, step=epoch_1000x)
                    except Exception:
                        pass


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None, data_loader_val=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    device = torch.device(f'cuda:{local_rank}')
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    use_condition_frames = getattr(args, 'use_condition_frames', False)
    if use_condition_frames:
        # Video frame generation mode
        save_folder = os.path.join(
            args.output_dir,
            "video-gen-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}-cond{}".format(
                model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
                model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], 
                args.num_images, args.img_size, getattr(args, 'max_condition_frames', 2)
            )
        )
    else:
        # ImageNet label generation mode (compatibility)
        save_folder = os.path.join(
            args.output_dir,
            "imagenet-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
                model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
                model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
            )
        )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict()) #备份当前模型参数
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict()) #用于存储ema模型参数
    
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict) #把字典中的参数加载到模型

    # 生成模式：支持条件帧生成或label生成
    use_condition_frames = getattr(args, 'use_condition_frames', False)
    
    if use_condition_frames:
        # 视频帧生成模式：从验证集获取条件帧和目标帧
        if data_loader_val is None:
            print("Warning: Validation dataloader not provided, skipping evaluation")
            return
        
        print("Generating video frames from validation set...")
        generated_folder = os.path.join(save_folder, "generated")
        target_folder = os.path.join(save_folder, "target")
        if misc.get_rank() == 0:
            os.makedirs(generated_folder, exist_ok=True)
            os.makedirs(target_folder, exist_ok=True)
        
        sample_count = 0
        action_losses = []  # Collect action losses for evaluation
        action_mse_list = []  # Collect action MSE for logging
        
        for batch_idx, batch in enumerate(data_loader_val):
            if sample_count >= args.num_images:
                break
            
            condition_frames = batch['condition_frames'].to(device, non_blocking=True)
            target_frame = batch['target_frame'].to(device, non_blocking=True)
            
            # Ensure condition_frames are in [-1, 1] range
            condition_frames = condition_frames.to(torch.float32)
            target_frame = target_frame.to(torch.float32)
            
            actual_batch_size = min(condition_frames.shape[0], args.num_images - sample_count)
            condition_frames = condition_frames[:actual_batch_size]
            target_frame = target_frame[:actual_batch_size]
            
            # Get action ground truth if available
            action_gt = None
            if 'action' in batch and batch['action'] is not None:
                action_gt = batch['action'].to(device, non_blocking=True)[:actual_batch_size]
            
            if batch_idx % 10 == 0:
                print(f"  Generating batch {batch_idx + 1}, samples {sample_count + 1}-{sample_count + actual_batch_size}")
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generated_frames = model_without_ddp.generate(condition_frames=condition_frames)
                
                # Evaluate action prediction if action_gt is available
                if action_gt is not None:
                    # Get action prediction from the model
                    # We need to forward pass with return_action=True
                    # Use a dummy noisy target_frame for action prediction (similar to training)
                    t_dummy = torch.zeros(actual_batch_size, device=device)
                    with torch.no_grad():
                        _, action_pred = model_without_ddp.net(
                            target_frame, t_dummy, 
                            condition_frames=condition_frames,
                            return_action=True
                        )
                    
                    # Calculate action MSE
                    action_mse = torch.nn.functional.mse_loss(action_pred, action_gt, reduction='mean')
                    action_losses.append(action_mse.item())
                    action_mse_list.append(action_mse.item())
            
            # Clamp and normalize generated frames
            generated_frames = torch.clamp(generated_frames, -1.0, 1.0)
            generated_frames = (generated_frames + 1) / 2
            generated_frames = torch.clamp(generated_frames, 0, 1).detach().cpu()
            
            # Normalize target frames
            target_frame_cpu = (target_frame + 1) / 2
            target_frame_cpu = torch.clamp(target_frame_cpu, 0, 1).detach().cpu()
            
            # Save images
            for b_id in range(actual_batch_size):
                img_id = sample_count + b_id
                if img_id >= args.num_images:
                    break
                
                # Save generated frame
                gen_img = generated_frames[b_id].numpy().transpose(1, 2, 0)
                gen_img = (gen_img * 255).astype(np.uint8)
                Image.fromarray(gen_img).save(
                    os.path.join(generated_folder, f'{img_id:05d}.png')
                )
                
                # Save target frame
                target_img = target_frame_cpu[b_id].numpy().transpose(1, 2, 0)
                target_img = (target_img * 255).astype(np.uint8)
                Image.fromarray(target_img).save(
                    os.path.join(target_folder, f'{img_id:05d}.png')
                )
            
            sample_count += actual_batch_size
        
        print(f"Generated {sample_count} pairs of images")
        
        # Log action evaluation metrics
        if action_losses and misc.get_rank() == 0:
            avg_action_mse = np.mean(action_losses)
            print(f"Action Prediction MSE: {avg_action_mse:.6f}")
            if log_writer is not None:
                log_writer.add_scalar('eval_action_mse', avg_action_mse, epoch)
            if getattr(args, '_use_wandb', False):
                try:
                    import wandb
                    wandb.log({'eval_action_mse': avg_action_mse}, step=epoch)
                except Exception:
                    pass
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Compute FID between generated and target images
        if log_writer is not None and HAS_TORCH_FIDELITY and misc.get_rank() == 0:
            print("Computing FID between generated and target images...")
            try:
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=generated_folder,
                    input2=target_folder,
                    cuda=True,
                    fid=True,
                    isc=False,
                    kid=False,
                    prc=False,
                    verbose=True,
                )
                fid = metrics_dict['frechet_inception_distance']
                postfix = "_video_cfg{}_res{}_cond{}".format(
                    model_without_ddp.cfg_scale, args.img_size, getattr(args, 'max_condition_frames', 2)
                )
                log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
                print(f"FID (generated vs target): {fid:.4f}")
                if getattr(args, '_use_wandb', False):
                    try:
                        import wandb
                        wandb.log({'eval_fid': fid}, step=epoch)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error computing FID: {e}")
        
        # Clean up temporary folders after FID calculation
        if misc.get_rank() == 0 and log_writer is not None:
            if not (HAS_TORCH_FIDELITY and log_writer is not None):
                # Only clean up if we're not using the images for FID
                shutil.rmtree(save_folder)
        
        return
    else:
        # ImageNet label生成模式（兼容性）
        class_num = args.class_num
        assert args.num_images % class_num == 0, "Number of images per class must be the same"
        class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

        for i in range(num_steps):
            print("Generation step {}/{}".format(i, num_steps))

            start_idx = world_size * batch_size * i + local_rank * batch_size
            end_idx = start_idx + batch_size
            labels_gen = class_label_gen_world[start_idx:end_idx]
            labels_gen = torch.Tensor(labels_gen).long().cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sampled_images = model_without_ddp.generate(labels=labels_gen)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)
            # 使用PIL保存图像（不需要OpenGL，适合无头服务器）
            Image.fromarray(gen_img).save(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict) #从ema切换回原始模型

    # compute FID and IS
    if log_writer is not None and HAS_TORCH_FIDELITY:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
    elif log_writer is not None and not HAS_TORCH_FIDELITY:
        print("Skipping FID/IS calculation: torch_fidelity not available")
        shutil.rmtree(save_folder)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_pusht_success_eval(model_without_ddp, args, epoch, log_writer=None):
    """
    Run PushT environment evaluation for success rate.
    Uses JitPushTPolicy and PushTImageRunner from UVA.
    """
    import pickle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uva_root = os.path.join(script_dir, "..", "unified_video_action")

    # Save JiT's util.misc before importing UVA (avoid module name collision)
    _saved_util_misc = sys.modules.get("util.misc")
    _saved_util = sys.modules.get("util")

    if uva_root not in sys.path:
        sys.path.insert(0, uva_root)

    from jit_push_policy import JitPushTPolicy
    from unified_video_action.env_runner.pusht_image_runner import PushTImageRunner

    # Restore JiT's util.misc
    if _saved_util_misc is not None:
        sys.modules["util.misc"] = _saved_util_misc
    if _saved_util is not None:
        sys.modules["util"] = _saved_util

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = getattr(args, "output_dir", "./output_dir")
    output_dir = os.path.join(output_dir, "pusht_eval")
    os.makedirs(output_dir, exist_ok=True)

    # Switch to EMA for eval
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _) in enumerate(model_without_ddp.named_parameters()):
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    normalizer = None
    normalizer_type = "none"
    np_path = getattr(args, "pusht_normalizer_path", "") or ""
    ds_path = getattr(args, "pusht_dataset_path", "") or ""
    if np_path and os.path.exists(np_path):
        with open(np_path, "rb") as f:
            normalizer = pickle.load(f)
        normalizer_type = "all"
    elif ds_path and os.path.exists(ds_path):
        from unified_video_action.dataset.pusht_image_dataset import PushTImageDataset
        from unified_video_action.model.common.normalizer import LinearNormalizer
        dataset = PushTImageDataset(dataset_path=ds_path, horizon=1, val_ratio=0)
        normalizer = dataset.get_normalizer(mode="limits")
        normalizer_type = "all"

    policy = JitPushTPolicy(
        denoiser=model_without_ddp,
        n_action_steps=8,
        max_condition_frames=getattr(args, "max_condition_frames", 2),
        img_size=getattr(args, "img_size", 256),
        action_dim=getattr(args, "action_dim", 2),
        normalizer=normalizer,
        normalizer_type=normalizer_type,
        device=device,
    )
    policy.eval()

    use_wandb_video = getattr(args, "pusht_wandb_video", False)
    n_test_vis = 4 if use_wandb_video else 0
    n_train_vis = 2 if use_wandb_video else 0

    env_runner = PushTImageRunner(
        output_dir=output_dir,
        n_train=6,
        n_train_vis=n_train_vis,
        train_start_seed=0,
        n_test=50,
        n_test_vis=n_test_vis,
        legacy_test=True,
        test_start_seed=100000,
        max_steps=300,
        n_obs_steps=16,
        n_action_steps=8,
        fps=10,
        render_size=96,
        past_action=False,
        fix_goal=True,
    )

    # Init wandb for video logging (like UVA)
    if use_wandb_video:
        try:
            import wandb
            if wandb.run is None:
                wandb.init(
                    project=getattr(args, "wandb_project", "jit-pusht"),
                    name=getattr(args, "wandb_run_name", None) or os.path.basename(getattr(args, "output_dir", "eval").rstrip("/")),
                    config={"epoch": epoch},
                )
        except ImportError:
            use_wandb_video = False

    print("Running PushT success rate evaluation...")
    runner_log = env_runner.run(policy)
    model_without_ddp.load_state_dict(model_state_dict)

    test_mean = runner_log.get("test/mean_score")
    train_mean = runner_log.get("train/mean_score")
    print(f"PushT test mean score: {test_mean:.4f}, train mean score: {train_mean:.4f}")

    if log_writer is not None and test_mean is not None:
        log_writer.add_scalar("pusht_test_mean_score", float(test_mean), epoch)
    if log_writer is not None and train_mean is not None:
        log_writer.add_scalar("pusht_train_mean_score", float(train_mean), epoch)

    if use_wandb_video:
        try:
            import wandb
            wandb.log(runner_log, step=epoch)
        except Exception:
            pass
