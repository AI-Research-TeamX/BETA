import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("verl.probe")
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


LABEL_SPECS = {
    "eq_type":       {"num_classes": 3, "task_filter": "nash_equilibrium"},
    "difficulty":    {"num_classes": 3, "task_filter": None},
    "dominance":     {"num_classes": 2, "task_filter": None},
    "br_direction":  {"num_classes": 5, "task_filter": "best_response"},
    "eq_uniqueness": {"num_classes": 2, "task_filter": "nash_equilibrium"},
}

CONCEPT_WEIGHTS = {
    "br_direction":  1.57,
    "eq_uniqueness": 1.37,
    "eq_type":       1.34,
    "dominance":     1.23,
    "difficulty":    1.11,
}


def is_probe_enabled():
    return os.environ.get("VERL_PROBE_ENABLED", "0") == "1"


def get_probe_lambda():
    return float(os.environ.get("VERL_PROBE_LAMBDA", "0.1"))


def get_probe_layer():
    return int(os.environ.get("VERL_PROBE_LAYER", "17"))


class ProbingHeads(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, spec["num_classes"])
            for name, spec in LABEL_SPECS.items()
        })

    def forward(self, hidden_states, labels_dict):
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        total_weight = 0.0
        # metric keys must be identical across all DP ranks and micro batches,
        # otherwise DataProto.concat asserts on mismatched metric dicts
        metrics = {f"probe/{name}_acc": float("nan") for name in self.heads}
        for name, head in self.heads.items():
            labels = labels_dict.get(name)
            if labels is None:
                continue
            valid = labels >= 0
            if not valid.any():
                continue
            logits = head(hidden_states[valid])
            loss = F.cross_entropy(logits, labels[valid])
            w = CONCEPT_WEIGHTS.get(name, 1.0)
            total_loss = total_loss + loss * w
            total_weight += w
            with torch.no_grad():
                acc = (logits.argmax(-1) == labels[valid]).float().mean()
                metrics[f"probe/{name}_acc"] = acc.item()
        if total_weight > 0:
            total_loss = total_loss / total_weight
        metrics["probe/total_loss"] = total_loss.item()
        return total_loss, metrics


class ProbeState:
    """Global singleton for probe state management across verl components."""
    _instance = None

    def __init__(self):
        self.enabled = False
        self.probing_heads = None
        self.probe_optimizer = None
        self.hidden_states = None
        self.concept_labels = None
        self.hook_handle = None
        self.probe_lambda = 0.1
        self.probe_layer = 17
        self.initialized = False

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _find_layers(self, model):
        """Unwrap FSDP/module wrappers and find transformer layers."""
        unwrapped = model
        for _ in range(10):
            if hasattr(unwrapped, '_fsdp_wrapped_module'):
                unwrapped = unwrapped._fsdp_wrapped_module
            elif hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module
            else:
                break

        if hasattr(unwrapped, 'model') and hasattr(unwrapped.model, 'layers'):
            return unwrapped.model.layers, unwrapped
        if hasattr(unwrapped, 'layers'):
            return unwrapped.layers, unwrapped
        return None, unwrapped

    def _get_hidden_size(self, model, unwrapped):
        """Get hidden_size from config, checking multiple locations."""
        for obj in [model, unwrapped]:
            if hasattr(obj, 'config') and hasattr(obj.config, 'hidden_size'):
                return obj.config.hidden_size
        return None

    def setup(self, model, device):
        if not is_probe_enabled() or self.initialized:
            return

        self.enabled = True
        self.probe_lambda = get_probe_lambda()
        self.probe_layer = get_probe_layer()

        layers, unwrapped = self._find_layers(model)
        if layers is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            logger.warning(f"[PROBE] Could not find model layers. Type: {type(unwrapped)}")
            self.enabled = False
            return

        hidden_size = self._get_hidden_size(model, unwrapped)
        if hidden_size is None:
            hidden_size = 2048
            logger.warning(f"[PROBE] Could not find hidden_size, defaulting to {hidden_size}")

        # seed probe-head init for reproducible multi-seed experiments
        probe_seed = int(os.environ.get("VERL_PROBE_SEED", "0"))
        torch.manual_seed(probe_seed)
        self.probing_heads = ProbingHeads(hidden_size).to(device).to(torch.bfloat16)
        self.probe_optimizer = torch.optim.Adam(
            self.probing_heads.parameters(), lr=1e-3
        )

        if self.probe_layer < len(layers):
            target_layer = layers[self.probe_layer]
        else:
            self.probe_layer = len(layers) // 2
            target_layer = layers[self.probe_layer]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hidden_states = output[0]
            else:
                self.hidden_states = output

        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        self.initialized = True
        logger.info(f"[PROBE] Initialized: layer={self.probe_layer}, lambda={self.probe_lambda}, "
                    f"hidden_size={hidden_size}, num_layers={len(layers)}")

    def compute_probe_loss(self, seq_info=None):
        """Compute probe loss from captured hidden states and concept labels.

        Args:
            seq_info: list of (prompt_len, response_len) tuples for packed sequences.
                      In packed mode (h.dim()==2), used to find sample boundaries
                      and extract prompt-only representations.
        """
        if not self.enabled or self.hidden_states is None or self.concept_labels is None:
            return None, {}

        h = self.hidden_states
        labels = self.concept_labels

        try:
            # remove-padding path packs all sequences into (1, total_nnz, hidden)
            if h.dim() == 3 and h.size(0) == 1 and seq_info is not None:
                total_tokens = sum(int(p) + int(r) for p, r in seq_info)
                if h.size(1) >= total_tokens and len(seq_info) > 1:
                    h = h.squeeze(0)

            if h.dim() == 2 and seq_info is not None and len(seq_info) > 0:
                reps = []
                offset = 0
                for prompt_len, response_len in seq_info:
                    prompt_len, response_len = int(prompt_len), int(response_len)
                    p_end = min(offset + prompt_len, h.shape[0])
                    if offset < p_end:
                        reps.append(h[offset:p_end].mean(dim=0))
                    offset += prompt_len + response_len
                if not reps:
                    return None, {}
                prompt_repr = torch.stack(reps)
            elif h.dim() == 3:
                if seq_info is not None and len(seq_info) > 0:
                    reps = []
                    for i in range(min(len(seq_info), h.shape[0])):
                        pl = min(int(seq_info[i][0]), h.shape[1])
                        reps.append(h[i, :pl].mean(dim=0))
                    prompt_repr = torch.stack(reps)
                else:
                    prompt_repr = h.mean(dim=1)
            else:
                return None, {}

            min_batch = min(prompt_repr.shape[0], min(len(v) for v in labels.values()))
            if min_batch == 0:
                return None, {}

            prompt_repr = prompt_repr[:min_batch]
            labels_trimmed = {k: v[:min_batch].to(prompt_repr.device) for k, v in labels.items()}

            probe_loss, metrics = self.probing_heads(prompt_repr, labels_trimmed)
            return probe_loss, metrics
        except Exception as e:
            logger.warning(f"[PROBE] compute_probe_loss error: {e}, h.shape={h.shape}")
            return None, {}

    def zero_grad(self):
        if self.enabled and self.probe_optimizer is not None:
            self.probe_optimizer.zero_grad()

    def step(self):
        if self.enabled and self.probe_optimizer is not None:
            self.probe_optimizer.step()

    def clear(self):
        self.hidden_states = None
        self.concept_labels = None
