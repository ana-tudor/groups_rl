import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from sklearn.cluster import KMeans

from studygroups_rl.infrastructure import pytorch_util as ptu
from studygroups_rl.policies.base_policy import BasePolicy

class SimpleDECPolicy(BasePolicy):
    def __init__(self, agent_params, **kwargs):
        super(SimpleDECPolicy, self).__init__(**kwargs)
        self.agent_params=agent_params
        self.input_dim = agent_params['input_dim']
        self.latent_dim = agent_params['latent_dim']
        self.k = agent_params['k']
        self.group_size = agent_params['group_size']
        self.learning_rate = agent_params['lr']
        self.ae_learning_rate = agent_params['ae_lr']
        self.ae = ptu.AE(
            input_size = agent_params['input_dim'],
            hidden_size = agent_params['hidden_dim_ae'],
            n_hidden= agent_params['n_hidden_layers_ae'],
            latent_size= agent_params['latent_dim'],
            activation= agent_params['layer_activation_ae'],
            output_activation= 'identity',
            dropout = agent_params['use_dropout'],
            dropout_rate = agent_params['dropout_rate'])
        self.ae.to(ptu.device)
        self.encoder = self.ae.encoder
        self.ae_loss = nn.MSELoss()
        self.mu = nn.Parameter(
            torch.zeros((self.k, self.latent_dim), requires_grad=True, device=ptu.device))
        self.ae_optimizer = optim.Adam(
            itertools.chain(self.ae.parameters()), #Include other parameters here?
            self.ae_learning_rate
        )
        self.dec_optimizer = optim.Adam(
            itertools.chain(self.encoder.parameters()), #Include other parameters here?
            self.learning_rate
        )
        self.mu_initialized = False
    
    def send_to_device(self):
        self.ae.to(ptu.device)
        self.mu.to(ptu.device)
        self.dec_optimizer.to(ptu.device)
        # self.

    def update_ae(self, obs: np.ndarray) -> dict:
        if len(obs.shape) > 1:
            observation = ptu.from_numpy(obs)
        else:
            observation = ptu.from_numpy(obs[None])

        ae_obs = self.ae(observation)
        loss = self.ae_loss(ae_obs, observation)
        
        self.ae_optimizer.zero_grad()
        loss.backward()
        self.ae_optimizer.step()

        return {"Autoencoder loss": loss.item()}

    def initialize_centers(self, obs):
        """
        Initialize cluster centers mu using k means in the latent space,
          before further iteration on cluster centers
        """
        # if len(obs.shape) > 1:
        #     observation = ptu.from_numpy(obs)
        # else:
        #     observation = ptu.from_numpy(obs[None])
        #Latent space representation of obs!
        z = self.encode(obs)
        kmeans = KMeans(n_clusters = self.k)
        kmeans.fit(ptu.to_numpy(z))

        clusters = kmeans.labels_
        assert torch.sum(self.mu) == 0, "Cluster centers must be zero upon initialization"
        centroids = ptu.from_numpy(kmeans.cluster_centers_)
        centroids.requires_grad=True
        centroids.to(ptu.device)
        self.mu = nn.Parameter(centroids).to(ptu.device)
        
        # self.mu = self.mu + ptu.from_numpy(kmeans.cluster_centers_)
        # self.dec_optimizer.add_param_group({"params":self.mu})
        return clusters

    def encode(self, observation: torch.FloatTensor):
        return self.encoder.forward(observation).to(ptu.device)

    def forward(self, observation: torch.FloatTensor):
        """
        Calculates latent space representation of observation, generates soft 
          centroid assignments
        """
        if (torch.sum(self.mu) == 0):
            self.initialize_centers(observation)
        # print(observation.size)
        n = observation.size()[0]
        # q_pre = torch.zeros((n, self.k),requires_grad=True, device=ptu.device)

        z = self.encode(observation)
        # print(ptu.device)
        # print(z.get_device())
        # print("mu",self.mu.get_device())

        q_pre2 = torch.subtract(z.unsqueeze(0), self.mu.unsqueeze(1))
        q_pre1 = torch.norm( q_pre2, p=2, dim=2).T
        q_pre = torch.pow( torch.add(1, torch.pow(q_pre1, 2)), -1)
        q = ((q_pre.T)/torch.sum(q_pre, dim=1)).T

        f = torch.sum(q, dim=0)
        p_pre = q**2/f
        p = ((p_pre.T)/torch.sum(p_pre, dim=1)).T

            
        return p, q, z

    def kl_loss(self, p: torch.FloatTensor, q:torch.FloatTensor) -> float:
        log_ratio = torch.log(torch.div(p, q))
        return torch.sum(p*log_ratio)

    def update_dec(self, obs: np.ndarray, eval= False,**kwargs) -> dict:
        """Update using DEC KL-divergence loss,
            return a dictionary of logging information."""
        # observation =torch.gather(
        #     ptu.from_numpy(obs), 
        #     dim=-1, 
        #     index=torch.arange(self.input_dim)
        # )
        observation = ptu.from_numpy(obs)

        stud_assignments, mus = self.get_action_nearest_mu(observation)
        self.mu = mus

        p, q, z = self.forward(observation)

        # stud_assignments, new_mus, p, q, z = self.get_action_nearest_mu(observation)
        # self.mu = 

        loss = self.kl_loss(p, q) # Only train relative to q

        if not eval:
            self.dec_optimizer.zero_grad()
            loss.backward()
            # print(self.mu.grad)
            self.dec_optimizer.step()

        return {"Policy KL Divergence":loss.detach().item()}

    def get_action_sample(self, obs:np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = ptu.from_numpy(obs)
        else:
            observation = ptu.from_numpy(obs[None])
        n = observation.size()[0]
        p, q, z = self.forward(observation)

        groups = torch.zeros((self.k, self.group_size))
        stud_assignments = torch.zeros((n,1))
        stud_dists = []

        for i in range(n):
            groups_available = torch.sum(groups>0, dim=1) < self.group_size
            probs_ = torch.where( groups_available, p[i], torch.zeros(self.k))
            probs = probs_/torch.sum(probs_)
            stud_dist = torch.distributions.Categorical(probs)
            stud_dists.append(stud_dist)
            stud_i_group = stud_dist.sample()
            stud_assignments[i] = stud_i_group
            groups[ stud_i_group, torch.sum(groups[stud_i_group]>0)] = i

        return stud_assignments

    def get_action_argmax(self, obs: np.ndarray) -> np.ndarray:
        p, q, z = self.forward(ptu.from_numpy(obs))
        return ptu.to_numpy(torch.argmax(p, dim=0))

    def get_action_nearest_mu(self, observation):
        p,q,z = self.forward(observation)
        n = z.size()[0]

        mu_freq = torch.sum(p, dim=0)
        mus_sorted = torch.argsort(mu_freq)
        # print(mus_sorted.)

        groups = torch.zeros((self.k,self.group_size,self.latent_dim)).to(ptu.device)
        # new_groups = torch.zeros((self.k, self.group_size,self.latent_dim))
        stud_assignments = torch.zeros((n,1)).to(ptu.device)

        for mu in mus_sorted:
            studs_mu = torch.argsort(p[:,mu], descending=True)[:self.group_size]
#             print(studs_mu)
            groups[mu] = z.detach()[studs_mu]
            for stud in studs_mu:
                stud_assignments[stud] = mu
                p[stud] = 0#10*torch.ones((1,self.k))
        new_mus = groups.mean(dim=1)
        # print(new_mus.size())
        return stud_assignments, new_mus

    def get_action(self, observation):
        p, q, z = self.forward(observation)
        stud_assignments, new_mus = self.get_action_nearest_mu(p, q, z)

        return stud_assignments


    def update_policy_AWAC(self, obs: np.ndarray, acs: np.ndarray, adv_n: np.ndarray, eval=False, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        if adv_n is None:
            assert False, "Did not pass advantages"
        if isinstance(obs, np.ndarray):
            observations = ptu.from_numpy(obs)
        if isinstance(acs, np.ndarray):
            actions = ptu.from_numpy(acs)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)
        
        torch.autograd.set_detect_anomaly(True)

        # observation =torch.gather(
        #     observations, 
        #     dim=-1, 
        #     index=torch.arange(self.agent_params["input_dim"])
        # )

        # Update the policy network utilizing AWAC update

        action_dist, q, z = self.forward(observations)
        
        # log_prob_actions = torch.log(torch.gather(action_dist, dim=1, index=actions))
        # 
        #Calculate log prob that each group of students gets assigned together as in the action
        # log_prob_actions = torch.zeros(observations.size()[0])
        # group_score = None
        scores = torch.zeros(1,requires_grad=True).to(ptu.device)


        for ac in set(acs):
            stud_probs = action_dist[acs==ac]#torch.gather(input=action_dist, dim=0, index=ptu.from_numpy(acs==ac).type(torch.int64))print(stud_probs.size())
            # print(stud_probs.size())
            group_score = torch.log(torch.max(torch.prod(stud_probs,dim=0)))
            # print(group_score.size())

            scores = scores+ torch.sum(group_score*adv_n[acs==ac])


        loss = torch.mean(-scores)/(observations.size()[0])
        # loss = - log_prob_actions * torch.exp(adv_n/self.lambda_awac)
        # actor_loss = loss.mean()

        if not eval:

            self.dec_optimizer.zero_grad()
            loss.backward()
            self.dec_optimizer.step()
        
        return {"Policy AC loss":loss.item()}


    def save(self, filepath: str):
        raise NotImplementedError
