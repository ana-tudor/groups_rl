from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch

from studygroups_rl.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, agent_params, **kwargs):
        super().__init__()
        self.ob_dim = agent_params['input_dim']
        # self.num_classes = agent_params['num_classes']
        self.k = agent_params['k']
        # self.discrete = agent_params['discrete']
        self.size = agent_params['critic_size']
        self.n_layers = agent_params['n_critic_layers']
        self.learning_rate = agent_params['critic_lr']
        self.group_size = agent_params['group_size']

        # critic parameters
        self.num_target_updates = agent_params['num_target_updates']
        self.num_grad_steps_per_target_update = agent_params['num_grad_steps_per_target_update']
        self.gamma = agent_params['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim*self.group_size,   #Concatenate all students 
            1,  # Regressing over group quality
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs):
        # Filter obs in each group
        return self.critic_network(obs).squeeze()

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs[:,:-1])
        predictions = self.critic_network(obs).squeeze()
        return ptu.to_numpy(predictions)

    def get_v_np(self, obs):
        obs = ptu.from_numpy(obs[:,:-1])
        predictions = self.forward(obs)
        return ptu.to_numpy(torch.mean(predictions))

    def get_v(self, obs):
        predictions = self.forward(obs[:,:-1])
        return torch.mean(predictions)


    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n, eval=False):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        
        ob_no = ptu.from_numpy(ob_no[:,:-1])
        ac_na = ptu.from_numpy(ac_na)
        # print(next_ob_no.shape)
        next_ob_no = ptu.from_numpy(next_ob_no[:,:-1])
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        target = None
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                next_v = self.forward(next_ob_no)
                # print("terminal_n", terminal_n.size())
                # print("reward_n", reward_n.size())
                target = reward_n.add( self.gamma * next_v * terminal_n.logical_not())
            
            curr_v = self.forward(ob_no)
            # print("currv",curr_v.size())
            # print("target",target.size())
            loss = self.loss(curr_v, target.detach())
            if not eval:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            

        return {"Training Loss":loss.item(), "Data q-values":ptu.to_numpy(curr_v.mean())}
