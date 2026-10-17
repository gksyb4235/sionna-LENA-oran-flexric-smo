"""Small genuine Double DQN: replay, online argmax, target evaluation."""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn

CHECKPOINT_SCHEMA = "drl-xapp-double-dqn-v1"


class QNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))

    def forward(self, state):
        return self.layers(state)


class DoubleDQN:
    def __init__(self, state_dim, action_count, seed, *, train=True,
                 gamma=0.97, batch_size=16, warmup=16, gradient_steps=4,
                 replay_capacity=20000, epsilon_decay=1000, learning_rate=3e-4):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(state_dim, action_count)
        self.target = copy.deepcopy(self.online).eval()
        self.training = train
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate) if train else None
        self.state_dim, self.action_count = state_dim, action_count
        self.gamma, self.batch_size, self.warmup = gamma, batch_size, warmup
        self.gradient_steps, self.replay_capacity = gradient_steps, replay_capacity
        self.epsilon_decay = epsilon_decay
        self.replay = []
        self.replay_cursor = 0
        self.transitions, self.gradient_updates, self.decisions = 0, 0, 0
        self.loaded_from = None

    @property
    def epsilon(self):
        return max(0.1, 0.9 - 0.8 * self.transitions / self.epsilon_decay) if self.training else 0.0

    def select(self, state, valid):
        if not valid.any():
            raise ValueError("No valid action")
        self.decisions += 1
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(valid))), True
        with torch.inference_mode():
            q = self.online(torch.as_tensor(state, dtype=torch.float32)).numpy()
        q[~valid] = -np.inf
        return int(np.argmax(q)), False

    def learn(self, state, action, value, next_state, terminal, next_valid):
        if not self.training:
            return []
        transition = (np.array(state, copy=True), int(action), float(value),
                      np.array(next_state, copy=True), bool(terminal), np.array(next_valid, copy=True))
        if len(self.replay) < self.replay_capacity:
            self.replay.append(transition)
        else:
            self.replay[self.replay_cursor] = transition
        self.replay_cursor = (self.replay_cursor + 1) % self.replay_capacity
        self.transitions += 1
        losses = []
        if len(self.replay) < max(self.warmup, self.batch_size):
            return losses
        for _ in range(self.gradient_steps):
            indexes = self.rng.choice(len(self.replay), self.batch_size, replace=False)
            batch = [self.replay[int(i)] for i in indexes]
            states, actions, rewards, next_states, dones, masks = zip(*batch)
            states = torch.as_tensor(np.stack(states), dtype=torch.float32)
            next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.long)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.bool)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
            with torch.no_grad():
                # Double DQN separates selection and evaluation networks.
                next_actions = self.online(next_states).masked_fill(~masks, -torch.inf).argmax(dim=1)
                next_q = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                targets = rewards + self.gamma * (~dones) * next_q
            predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predictions, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
            self.optimizer.step()
            self.gradient_updates += 1
            if self.gradient_updates % 100 == 0:
                self.target.load_state_dict(self.online.state_dict())
            losses.append(float(loss.item()))
        return losses

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": CHECKPOINT_SCHEMA, "state_dim": self.state_dim,
            "action_count": self.action_count, "metadata": metadata,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "replay": self.replay, "replay_cursor": self.replay_cursor,
            "rng_state": self.rng.bit_generator.state,
            "transitions": self.transitions, "gradient_updates": self.gradient_updates,
            "decisions": self.decisions,
            "hyperparameters": {k: getattr(self, k) for k in ("gamma", "batch_size", "warmup",
                "gradient_steps", "replay_capacity", "epsilon_decay")}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(self, path, expected_metadata):
        # Only locally produced trusted experiment checkpoints should be loaded.
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload["schema"] != CHECKPOINT_SCHEMA or payload["state_dim"] != self.state_dim or payload["action_count"] != self.action_count:
            raise ValueError("Checkpoint architecture/schema mismatch")
        if payload["metadata"] != expected_metadata:
            raise ValueError("Checkpoint action space, app, state or reward mismatch")
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.transitions = payload["transitions"]
        self.gradient_updates = payload["gradient_updates"]
        self.decisions = payload["decisions"]
        if self.training:
            if payload["optimizer"] is None:
                raise ValueError("Cannot resume training from a checkpoint without optimizer")
            self.optimizer.load_state_dict(payload["optimizer"])
            self.replay = payload["replay"]
            self.replay_cursor = payload["replay_cursor"]
            self.rng.bit_generator.state = payload["rng_state"]
            for key, value in payload["hyperparameters"].items():
                setattr(self, key, value)
        else:
            self.online.eval()
            self.target.eval()
        self.loaded_from = str(Path(path).resolve())

    def stats(self):
        return {"trainable_parameters": sum(p.numel() for p in self.online.parameters()),
            "transitions_total": self.transitions, "gradient_updates_total": self.gradient_updates,
            "replay_size": len(self.replay), "epsilon": self.epsilon,
            "loaded_from": self.loaded_from, "learning_enabled": self.training}
