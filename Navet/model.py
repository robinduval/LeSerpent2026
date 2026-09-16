# Model DQN baseline Loeber : Linear_QNet(11, 256, 3) + QTrainer sans target network.
import torch
import torch.nn as nn
import torch.optim as optim


class Linear_QNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class QTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9, target_model=None):
        self.model = model
        self.target_model = target_model  # Fix 1/4 : si None = baseline instable
        self.gamma = gamma  # flaw volontaire : myope, 0.97+ testé Étape 2
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.last_loss = 0.0

    def sync_target(self):
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done, truncated=False):
        import numpy as np
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,) if isinstance(done, bool) else done
            truncated = (truncated,) if isinstance(truncated, bool) else truncated

        pred = self.model(state)
        target = pred.clone().detach()
        qnet = self.target_model if self.target_model is not None else self.model
        for i in range(len(done)):
            terminated = bool(done[i]) and not bool(truncated[i])
            q_new = reward[i]
            if not terminated:
                # Fix 4/4 : famine tronquée => on bootstrappe, mort => pas de futur
                q_new = reward[i] + self.gamma * torch.max(qnet(next_state[i]).detach())
            target[i][torch.argmax(action[i]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.item()
        return self.last_loss
