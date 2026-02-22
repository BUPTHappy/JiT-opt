import torch
import torch.nn as nn
from model_jit import JiT_models

#完整的扩散模型训练和生成系统

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        max_condition_frames = getattr(args, 'max_condition_frames', 2)
        text_latent_dim = getattr(args, 'text_latent_dim', 512)
        use_text_condition = getattr(args, 'use_text_condition', False)
        action_dim = getattr(args, 'action_dim', 10)  # Default 10 for UMI, can be set to 2 for pushT
        action_horizon = getattr(args, 'action_horizon', 1)  # Number of future action steps

        self.net = JiT_models[args.model](  #通过args.model选择模型架构
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            max_condition_frames=max_condition_frames,
            text_latent_dim=text_latent_dim,
            use_text_condition=use_text_condition,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.max_condition_frames = max_condition_frames
        self.use_text_condition = use_text_condition
        
        self.label_drop_prob = args.label_drop_prob #标签drop概率
        self.text_drop_prob = getattr(args, 'text_drop_prob', 0.1)
        self.condition_drop_prob = getattr(args, 'condition_drop_prob', 0.1)  # 条件帧drop概率（用于CFG训练）
        
        # action prediction weight (for multi-task learning)
        self.action_loss_weight = getattr(args, 'action_loss_weight', 0.1)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)
        self.action_only_loss = getattr(args, 'action_only_loss', False)  # action-only fine-tune, all params trainable

        self.P_mean = args.P_mean #时间步采样的均值
        self.P_std = args.P_std #时间步采样的标准差
        self.t_eps = args.t_eps #时间步的最小值
        self.noise_scale = args.noise_scale  #噪声缩放因子（控制添加到图像中的噪声强度）

        # ema 对模型参数做指数滑动平均
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg #CFG缩放因子
        self.cfg_interval = (args.interval_min, args.interval_max) #CFG间隔

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out
    
    def drop_text_latents(self, text_latents):
        """Drop text latents for CFG training"""
        if text_latents is None:
            return None, None  # 返回两个None
        drop = torch.rand(text_latents.shape[0], device=text_latents.device) < self.text_drop_prob
        return text_latents, drop
    

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels=None, condition_frames=None, text_latents=None, action_gt=None): #JIT论文中提到的选用的理论公式
        # handle labels (for compatibility)
        labels_dropped = None
        if labels is not None:
            labels_dropped = self.drop_labels(labels) if self.training else labels
        
        # handle text latents (for CFG)
        text_latents_dropped = None
        if text_latents is not None and self.use_text_condition:
            if self.training:
                text_latents_clean, drop_mask = self.drop_text_latents(text_latents)
                if drop_mask is not None:  # 添加检查
                    # 对于drop的样本，设置为零向量（无条件）
                    text_latents_dropped = torch.where(
                        drop_mask.unsqueeze(-1).expand_as(text_latents),
                        torch.zeros_like(text_latents),
                        text_latents
                    )
                    # 如果全部drop，设置为None
                    if drop_mask.all():
                        text_latents_dropped = None
                else:
                    text_latents_dropped = text_latents
            else:
                text_latents_dropped = text_latents
        
        # handle condition_frames (for CFG training)
        # 注意：condition_frames的dropout需要在batch级别处理
        # 为了简化，在训练时随机将整个batch的condition_frames设置为None
        # 这样模型会学习：50%的时间有条件，50%的时间无条件
        # 但是，如果freeze_backbone=True或action_only_loss=True（只训练/微调action），我们不应该drop condition_frames
        # 因为action需要从condition_frames中提取
        condition_frames_dropped = condition_frames
        if condition_frames is not None and self.training and not self.freeze_backbone and not self.action_only_loss:
            # 随机drop整个batch的condition_frames（用于CFG训练）
            # 但是当freeze_backbone=True时，不drop，因为需要condition_frames来提取action
            if torch.rand(1, device=condition_frames.device).item() < self.condition_drop_prob:
                condition_frames_dropped = None

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e #增加噪声
        v = (x - z) / (1 - t).clamp_min(self.t_eps) #计算速度场

        # Forward pass with optional action prediction
        return_action = (action_gt is not None)
        net_output = self.net(z, t.flatten(), y=labels_dropped, 
                             condition_frames=condition_frames_dropped, 
                             text_latents=text_latents_dropped,
                             return_action=return_action)
        
        if return_action:
            x_pred, action_pred = net_output
        else:
            x_pred = net_output
        
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # Image generation loss (L2)
        # Skip when backbone frozen or action-only mode (saves computation)
        loss_image = None
        if not self.freeze_backbone and not self.action_only_loss:
            loss_image = (v - v_pred) ** 2
            loss_image = loss_image.mean(dim=(1, 2, 3)).mean()

        # Action prediction loss (if action_gt is provided)
        loss_action = None
        if action_gt is not None:
            loss_action = torch.nn.functional.mse_loss(action_pred, action_gt)
        
        # DEBUG: print loss components on first forward pass
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[DEBUG denoiser] freeze_backbone={self.freeze_backbone}, "
                  f"action_only_loss={self.action_only_loss}")
            print(f"[DEBUG denoiser] loss_image={loss_image}, loss_action={loss_action}")
            if action_gt is not None:
                print(f"[DEBUG denoiser] action_pred min={action_pred.min().item():.4f}, "
                      f"max={action_pred.max().item():.4f}, "
                      f"mean={action_pred.mean().item():.4f}")
                print(f"[DEBUG denoiser] action_gt   min={action_gt.min().item():.4f}, "
                      f"max={action_gt.max().item():.4f}, "
                      f"mean={action_gt.mean().item():.4f}")

        # Combined loss
        if self.freeze_backbone or self.action_only_loss:
            # Only use action loss: freeze_backbone=只训action_head, action_only_loss=全模型微调
            if loss_action is not None:
                loss = loss_action
            else:
                raise ValueError("action-only mode but no action_gt provided")
        else:
            # Normal training: combine both losses
            if loss_image is not None and loss_action is not None:
                loss = loss_image + self.action_loss_weight * loss_action
            elif loss_image is not None:
                loss = loss_image
            elif loss_action is not None:
                loss = loss_action
            else:
                raise ValueError("No loss to compute: both image and action losses are None")

        # DEBUG: print final loss on first forward
        if not hasattr(self, '_debug_loss_printed'):
            self._debug_loss_printed = True
            print(f"[DEBUG denoiser] final loss={loss.item():.4f}")

        return loss

    @torch.no_grad()
    def generate(self, condition_frames=None, text_latents=None, labels=None):
        if condition_frames is not None:
            device = condition_frames.device
            bsz = condition_frames.shape[0]
        elif labels is not None:
            device = labels.device
            bsz = labels.size(0)
        else:
            raise ValueError("Either condition_frames or labels must be provided")

        #1. #初始化为纯噪声
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device) #3个通道，图像大小，设备
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        #采样方法选择
        if self.method == "euler": # 一阶精度
            stepper = self._euler_step
        elif self.method == "heun": # 二阶精度（更准确但慢2倍）
            stepper = self._heun_step
        else:
            raise NotImplementedError
    
        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, condition_frames=condition_frames, 
                       text_latents=text_latents, labels=labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], 
                           condition_frames=condition_frames,
                           text_latents=text_latents, labels=labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, condition_frames=None, text_latents=None, labels=None): # 告诉我"从当前状态 z 往哪个方向走"，计算速度场
        # conditional —— 知道要生成什么，有label
        x_cond = self.net(z, t.flatten(), y=labels, 
                         condition_frames=condition_frames,
                         text_latents=text_latents)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional prediction
        if text_latents is not None and self.use_text_condition:
            # CFG with text: unconditional = text_latents=None
            x_uncond = self.net(z, t.flatten(), y=labels,
                               condition_frames=condition_frames,
                               text_latents=None)
        elif labels is not None:
            # CFG with labels: unconditional = null class
            x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes),
                               condition_frames=condition_frames,
                               text_latents=None)
        elif condition_frames is not None:
            # CFG with condition_frames: unconditional = condition_frames=None
            x_uncond = self.net(z, t.flatten(), y=labels,
                               condition_frames=None,
                               text_latents=None)
        else:
            # No CFG, just use conditional
            return v_cond
        
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0) #判断是否在CFG间隔内，在则使用CFG缩放因子，否则使用1.0

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, condition_frames=None, text_latents=None, labels=None):
        v_pred = self._forward_sample(z, t, condition_frames=condition_frames,
                                     text_latents=text_latents, labels=labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, condition_frames=None, text_latents=None, labels=None):
        v_pred_t = self._forward_sample(z, t, condition_frames=condition_frames,
                                       text_latents=text_latents, labels=labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, 
                                            condition_frames=condition_frames,
                                            text_latents=text_latents, labels=labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
