'''
Based on https://github.com/davda54/sam
'''

import torch
from torch.optim import AdamW
import torch.nn as nn
from typing import Iterable

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        初始化 SAM 优化器

        Args:
            params: 训练参数列表
            base_optimizer: 基础优化器类（如 torch.optim.SGD 或 torch.optim.Adam）
            rho: 控制扰动幅度的超参数
            adaptive: 是否使用自适应扰动（参数大小相关）
            **kwargs: 传递给基础优化器的参数
        """

        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # 初始化默认参数
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

         # 初始化基础优化器
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups # 绑定参数组（确保一致）

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        第一步：在原始参数点 w 上，计算梯度，并执行一步“上山”扰动（找到 sharp 位置）

        Args:
            zero_grad: 是否在最后清空梯度
        """
        grad_norm = self._grad_norm() # 计算整体 L2 梯度范数

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) # 根据 rho 和范数得到扰动缩放因子

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone() # 保存原始参数以便恢复

                #  e(w): 计算扰动方向
                # 若为 adaptive 模式，扰动与参数幅度成正比
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"  # 添加扰动到参数，得到 w + e(w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        第二步：从原始参数点 w 恢复，然后使用在 w + e(w) 计算出的梯度执行实际更新

        Args:
            zero_grad: 是否在最后清空梯度
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)" # 恢复为原始参数 w

        self.base_optimizer.step()  # do the actual "sharpness-aware" update # 用 w + e(w) 得到的梯度更新原始参数 w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        SAM 的核心 step 函数，调用前必须传入 closure 函数，完成完整的：
            - w → w + e(w)
            - 计算梯度 @ w + e(w)
            - 更新参数 @ w

        Args:
            closure: 闭包函数，需封装完整的前向 + 反向传播过程
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # 重新启用 grad 模式以确保 closure 能反向传播
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)   # 第一步：添加扰动并清除梯度
        closure()  # 第二步：在 w + e(w) 上计算梯度
        self.second_step() # 第三步：恢复 w，并用上一步梯度执行更新

    def _grad_norm(self):
        """
        计算当前参数的 L2 范数，用于归一化扰动方向
        """
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def disable_running_stats(model):
    """
    在 first_step 前调用，禁用 BatchNorm 的统计行为
    """
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    """
    在恢复模型（second_step 后）调用，恢复 BatchNorm 的 momentum
    """
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


        
class SAMAdamW(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, rho=0.05):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']

            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads
            torch._foreach_mul_(epsilon, rho / grad_norm)

            torch._foreach_add_(params_with_grads, epsilon)
            closure()
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss