from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpNormCertificate:
    model_name: str
    sigma_hat: float
    num_iter: int
    seed: int
    weight_shape: tuple[int, int]
    device: str
    dtype: str
    sigma_history: list[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sigma_hat": float(self.sigma_hat),
            "num_iter": int(self.num_iter),
            "seed": int(self.seed),
            "weight_shape": [int(self.weight_shape[0]), int(self.weight_shape[1])],
            "device": self.device,
            "dtype": self.dtype,
            "sigma_history": [float(x) for x in self.sigma_history],
        }


def _power_iteration_spectral_norm(
    weight, *, num_iter: int, seed: int, eps: float = 1e-12
) -> tuple[float, list[float]]:
    # weight: torch.Tensor [V, D]
    import torch

    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape={tuple(weight.shape)}")
    vocab, dim = int(weight.shape[0]), int(weight.shape[1])
    if vocab <= 0 or dim <= 0:
        raise ValueError(f"Invalid weight shape: {tuple(weight.shape)}")

    g = torch.Generator(device=weight.device)
    g.manual_seed(int(seed))

    v = torch.randn((dim,), generator=g, device=weight.device, dtype=weight.dtype)
    v = v / (v.norm() + eps)

    sigmas: list[float] = []
    for _ in range(int(num_iter)):
        u = weight @ v  # [V]
        sigma = float(u.norm().item())
        sigmas.append(sigma)
        u = u / (u.norm() + eps)
        v = weight.T @ u  # [D]
        v = v / (v.norm() + eps)

    return (sigmas[-1] if sigmas else 0.0), sigmas


def estimate_output_opnorm_hf(
    *,
    model_adapter,
    model_name: str,
    num_iter: int = 50,
    seed: int = 0,
    device: str | None = None,
    dtype: str | None = None,
) -> OpNormCertificate:
    """Estimate ||W_out||_op (spectral norm) for a HF CausalLM output projection.

    Notes:
    - This is meant for *offline* auditing / calibration (v38): compute once and cache.
    - The estimate uses power iteration on W (via matvec Wv and W^T u), which is
      compute-heavy for large vocabularies. Prefer GPU and float32 if possible.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError("op-norm estimation requires torch (pip install -e '.[hf]')") from e

    weight = model_adapter.get_output_weight()
    if device is not None:
        weight = weight.to(device)
    if dtype is not None:
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype}")
        weight = weight.to(dtype=torch_dtype)

    sigma_hat, history = _power_iteration_spectral_norm(weight, num_iter=num_iter, seed=seed)

    return OpNormCertificate(
        model_name=str(model_name),
        sigma_hat=float(sigma_hat),
        num_iter=int(num_iter),
        seed=int(seed),
        weight_shape=(int(weight.shape[0]), int(weight.shape[1])),
        device=str(weight.device),
        dtype=str(weight.dtype).replace("torch.", ""),
        sigma_history=history,
    )

