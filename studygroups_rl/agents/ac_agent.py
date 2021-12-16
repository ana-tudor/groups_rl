from collections import OrderedDict

from studygroups_rl.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
# from studygroups_rl.critics.bootstrapped_continuous_critic import \
#     CQLCritic
from studygroups_rl.infrastructure.replay_buffer import ReplayBuffer
from studygroups_rl.infrastructure.utils import *
from studygroups_rl.policies.dec_policy import SimpleDECPolicy
from .base_agent import BaseAgent
import torch

from studygroups_rl.infrastructure import pytorch_util as ptu

class AWACAgent(BaseAgent):
    def __init__(self, agent_params):
        super(AWACAgent, self).__init__()

        # self.data = data
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.num_param_updates = 0
        self.target_update_freq = agent_params['target_update_freq']
        self.input_dim = agent_params['input_dim']
        self.group_size = agent_params['group_size']

        self.actor = SimpleDECPolicy(self.agent_params)
        self.critic = BootstrappedContinuousCritic(self.agent_params)
        self.t = 0

        # self.critic = CQLCritic(self.agent_params)


        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, eval=False):
        # TODO Implement the following pseudocode:

        #Actor should be loaded with pre-trained auto-encoder

        #Then initiate actor-critic learning
        # print(ob_no.size())
        torch.autograd.set_detect_anomaly(True)

        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            crit_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n, eval)


        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            #Maybe self.actor.forward, then estimate advantage for that new action, 
            # and train the actor 
            # print(i)
            if i%2==0:
                ac_loss1 = self.actor.update_dec(ob_no[:,:self.input_dim], eval) #observations, actions, adv_n
            else:
                ac_loss2 = self.actor.update_policy_AWAC(ob_no[:,:self.input_dim], ac_na, advantage, eval)

        loss = OrderedDict()
        loss['Critic Loss'] = crit_loss['Training Loss']
        loss['Data q-values'] = crit_loss['Data q-values']
        if i%2==0:
            loss['Actor_Loss_KL'] = ac_loss1["Policy KL Divergence"]
        else:
            loss['Actor_Loss_'] = ac_loss2["Policy AC loss"]
        # loss['Reward']

        
        # Target Networks #
        # if self.num_param_updates % self.target_update_freq == 0:
        #     # Update the CQL critic target network
        #     self.critic.update_target_network()


        self.num_param_updates += 1
        self.t +=1

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        # print("obs size",ob_no.size())

        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        curr_v = self.critic.get_v(ob_no)
        # print("curr v", curr_v.size())
        next_v = self.critic.get_v(next_ob_no)
        q_est = re_n.add( self.gamma * next_v * terminal_n.logical_not())
        adv_n = ptu.to_numpy(q_est - curr_v)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)

    def eval(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        results = {}
        
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        results['Eval KL Loss'] = self.actor.update_dec(ob_no[:,:self.input_dim], eval=True)["Policy KL Divergence"] #observations, actions, adv_n
        results['Eval AC Loss'] = self.actor.update_policy_AWAC(ob_no[:,:self.input_dim], ac_na, advantage, eval=True)["Policy AC loss"]

        results['Eval value critic loss'] =  self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n,eval=True)['Training Loss']

        pred_groups, new_mus = self.actor.get_action_nearest_mu(ptu.from_numpy(ob_no[:,:self.input_dim]))
        # pred_groups = ptu.to_numpy(pred_groups)
        obs_new = self.permute_state(ob_no, pred_groups)
        
        results['Eval loss for actor predicted groups'] = self.critic.get_v(obs_new)

        return results

    def permute_state(self, ob, acs):
        big_ob = torch.zeros((ob.size()[0], self.group_size*self.input_dim+1))
        for ac in set(ptu.to_numpy(acs)):
            idxs = acs==ac
            group_obs = ob[idxs]
            group_preds = torch.zeros((self.group_size, self.group_size*self.input_dim+1))
            for i in range(self.group_size):
                group_preds[i, :self.input_dim] = group_obs[i]
                # other_mems = np.arange(self.group_size)[np.arange(self.group_size) != i]
                group_preds[i, self.input_dim:-1] = torch.flatten(group_obs[torch.arange(self.group_size) != i])
                group_preds[i, -1] = ac

            big_ob[idxs] = group_preds
        return big_ob

        