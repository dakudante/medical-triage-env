from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from models import ResourceRequest, TriageAction


RESOURCE_KEYS: List[str] = [
    "icu_bed", "er_bed", "resus_bay", "ventilator",
    "ct_scanner", "or_room", "cardiac_monitor", "cath_lab",
]


@dataclass
class PolicyOutput:
    action: TriageAction
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    debug: Dict[str, object]


class TriagePolicyNet(nn.Module):
    def __init__(self, input_dim: int, departments: List[str], hidden_dim: int = 128):
        super().__init__()
        self.departments = list(departments)
        self.routing_choices = ["admit", "wait", "reroute"]

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.esi_head = nn.Linear(hidden_dim, 5)
        self.dept_head = nn.Linear(hidden_dim, len(self.departments))
        self.routing_head = nn.Linear(hidden_dim, len(self.routing_choices))
        self.resource_head = nn.Linear(hidden_dim, len(RESOURCE_KEYS))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        hidden = self.backbone(x)
        return {
            "esi_logits": self.esi_head(hidden),
            "dept_logits": self.dept_head(hidden),
            "routing_logits": self.routing_head(hidden),
            "resource_logits": self.resource_head(hidden),
            "value": self.value_head(hidden).squeeze(-1),
        }

    def sample_action(self, x: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        outputs = self.forward(x)
        esi_dist = Categorical(logits=outputs["esi_logits"])
        dept_dist = Categorical(logits=outputs["dept_logits"])
        routing_dist = Categorical(logits=outputs["routing_logits"])
        resource_dist = Bernoulli(logits=outputs["resource_logits"])

        if deterministic:
            esi_idx = torch.argmax(outputs["esi_logits"], dim=-1)
            dept_idx = torch.argmax(outputs["dept_logits"], dim=-1)
            routing_idx = torch.argmax(outputs["routing_logits"], dim=-1)
            resource_bits = (torch.sigmoid(outputs["resource_logits"]) >= 0.5).float()
        else:
            esi_idx = esi_dist.sample()
            dept_idx = dept_dist.sample()
            routing_idx = routing_dist.sample()
            resource_bits = resource_dist.sample()

        log_prob = (
            esi_dist.log_prob(esi_idx)
            + dept_dist.log_prob(dept_idx)
            + routing_dist.log_prob(routing_idx)
            + resource_dist.log_prob(resource_bits).sum(dim=-1)
        )
        entropy = (
            esi_dist.entropy()
            + dept_dist.entropy()
            + routing_dist.entropy()
            + resource_dist.entropy().sum(dim=-1)
        )

        resource_payload = {
            key: bool(resource_bits[0, i].item() if resource_bits.dim() == 2 else resource_bits[i].item())
            for i, key in enumerate(RESOURCE_KEYS)
        }
        esi_level = int((esi_idx.item() if esi_idx.dim() == 0 else esi_idx[0].item()) + 1)
        dept = self.departments[dept_idx.item() if dept_idx.dim() == 0 else dept_idx[0].item()]
        routing = self.routing_choices[routing_idx.item() if routing_idx.dim() == 0 else routing_idx[0].item()]

        action = TriageAction(
            esi_level=esi_level,
            department=dept,
            routing_decision=routing,
            resource_request=ResourceRequest(**resource_payload),
            reasoning=f"PPO-lite policy chose ESI-{esi_level}, {dept}, routing={routing}.",
        )
        return PolicyOutput(
            action=action,
            log_prob=log_prob.squeeze(0),
            entropy=entropy.squeeze(0),
            value=outputs["value"].squeeze(0),
            debug={
                "resource_bits": resource_payload,
                "routing": routing,
                "department": dept,
                "esi_level": esi_level,
            },
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        esi_actions: torch.Tensor,
        dept_actions: torch.Tensor,
        routing_actions: torch.Tensor,
        resource_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward(states)
        esi_dist = Categorical(logits=outputs["esi_logits"])
        dept_dist = Categorical(logits=outputs["dept_logits"])
        routing_dist = Categorical(logits=outputs["routing_logits"])
        resource_dist = Bernoulli(logits=outputs["resource_logits"])

        log_prob = (
            esi_dist.log_prob(esi_actions)
            + dept_dist.log_prob(dept_actions)
            + routing_dist.log_prob(routing_actions)
            + resource_dist.log_prob(resource_actions).sum(dim=-1)
        )
        entropy = (
            esi_dist.entropy()
            + dept_dist.entropy()
            + routing_dist.entropy()
            + resource_dist.entropy().sum(dim=-1)
        )
        values = outputs["value"]
        return log_prob, entropy, values
