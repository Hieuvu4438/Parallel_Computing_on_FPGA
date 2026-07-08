"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

SAM seeks parameters that lie in neighborhoods with uniformly low loss
(flat minima), which empirically generalize better than sharp minima.

This implementation wraps any base optimizer (Adam, SGD, etc.) and
adds the perturbation step. Usage is identical to a regular optimizer
except for the two-step update in the training loop.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
           Improving Generalization", ICLR 2021

NOTE: This only affects training. Evaluation/inference is unchanged.
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            base_optimizer: The base optimizer class (e.g., torch.optim.Adam)
            rho: Size of the neighborhood to search for flat minima.
                 Typical: 0.01-0.1. Larger = more regularization.
            adaptive: If True, uses ASAM (adaptive scaling by parameter magnitude)
            **kwargs: Arguments passed to base_optimizer constructor
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.adaptive = adaptive

        # Store kwargs for base optimizer construction
        self.base_optimizer_cls = base_optimizer
        self.base_optimizer_kwargs = kwargs

        # Will be initialized in first step
        self._base_optimizer = None
        self.param_groups_list = []

    @property
    def param_groups(self):
        if self._base_optimizer is not None:
            return self._base_optimizer.param_groups
        return self.param_groups_list

    def setup(self, params):
        """Initialize with actual parameters. Call once before training."""
        self._base_optimizer = self.base_optimizer_cls(params, **self.base_optimizer_kwargs)
        self.param_groups_list = self._base_optimizer.param_groups
        self.state = {}  # Initialize state dict for storing epsilon

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """Compute and apply perturbation epsilon."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.adaptive:
                    eps = p.grad * scale * (p.abs() + 1e-8)
                else:
                    eps = p.grad * scale
                # Store epsilon for rollback
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]['eps'] = eps
                p.add_(eps)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """Rollback perturbation and apply the actual optimizer step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p not in self.state:
                    continue
                # Undo the perturbation
                p.sub_(self.state[p]['eps'])

        # Standard optimizer step with gradients computed at perturbed point
        self._base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self):
        """
        For compatibility: if called as a regular optimizer (without first/second step),
        just do a normal optimizer step.
        """
        self._base_optimizer.step()

    def zero_grad(self):
        if self._base_optimizer is not None:
            self._base_optimizer.zero_grad()

    def state_dict(self):
        if self._base_optimizer is not None:
            return self._base_optimizer.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._base_optimizer is not None:
            self._base_optimizer.load_state_dict(state_dict)

    def _grad_norm(self):
        """Compute the gradient norm across all parameters."""
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if self.adaptive:
                        norms.append((p.grad * (p.abs() + 1e-8)).norm())
                    else:
                        norms.append(p.grad.norm())
        if len(norms) == 0:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms))

    def __repr__(self):
        return f"SAM(rho={self.rho}, adaptive={self.adaptive}, base={self.base_optimizer_cls.__name__})"
