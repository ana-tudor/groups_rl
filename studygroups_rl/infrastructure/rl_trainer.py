from collections import OrderedDict
import pickle
import os
import sys
import time

# import gym
# from gym import wrappers
import numpy as np
import torch
from studygroups_rl.infrastructure import pytorch_util as ptu

from studygroups_rl.infrastructure import utils
from studygroups_rl.infrastructure.logger import Logger

from studygroups_rl.agents.ac_agent import AWACAgent
# from studygroups_rl.infrastructure.dqn_utils import (
#         get_wrapper_by_name,
#         register_custom_envs,
# )

# how many rollouts to save as videos to tensorboard
# MAX_NVIDEO = 2
# MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        #Load the data

        with open("permuted_data.pkl", "rb") as file:
            self.data = pickle.load(file)#{"Step0":cl_perm0, "Step1":cl_perm1, "Step2":cl_perm2}, file)
        
        len_data = len(self.data['Step0'])
        indices = np.random.permutation(np.arange(len(self.data['Step0'])))

        self.eval_indices = indices[:int(.05*len_data)]
        self.train_indices = indices[int(.05*len_data):]

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          buffer_name=None,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        
        with open("best_ae_trained.pkl", "rb") as file:
            best_model = pickle.load(file)
        # dec_actor.ae=best_model.ae
        # dec_actor.ae.to(ptu.device)
        # dec_actor.mu.to(ptu.device)
        self.agent.actor.ae = best_model.ae
        self.agent.actor.ae.to(ptu.device)

        #Load train, eval trajectories here


        print_period = 1#1000 if isinstance(self.agent, AWACAgent) else 1
        print("Iterations planned:", n_iter)
        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False


            paths = self.collect_training_trajectories(self.train_indices, 5)

            # add collected data to replay buffer
            # if isinstance(self.agent, AWACAgent):
            #     if (not self.agent.offline_exploitation) or (self.agent.t <= self.agent.num_exploration_steps):
            # self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")

            all_logs = self.train_agent(paths)

            # Log densities and output trajectories
            # if isinstance(self.agent, AWACAgent) and (itr % print_period == 0):
            #     self.dump_density_graphs(itr)

            # log/save
            if self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_dqn_logging(all_logs)
                #self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, idxs, num_transitions_to_sample):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # Indices


        paths = []
        # transitions = ["Step0", "Step1", "Step2"]
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        step = 0

        # print(ids)
        
        ids1 = np.random.permutation(idxs)
        for i in ids1:
            tr0 = self.data["Step0"][i]
            ids2 = np.random.permutation(idxs)
            for j in ids2:
                tr1 = self.data["Step1"][j]
                ids3 = np.random.permutation(idxs)
                for k in ids3:
                    tr2 = self.data["Step2"][k]
                    obs = [tr0["obs"], tr1["obs"], tr2["obs"]]
                    acs = [tr0["next_obs"][:,-1], tr1["next_obs"][:,-1], tr2["next_obs"][:,-1]]
                    # acs.append(tr["next_obs"][:,-1])
                    next_obs = [tr0["next_obs"], tr1["next_obs"], tr2["next_obs"]]

                    rewards = [tr0["df"]["reward_ind"].to_numpy(), 
                                tr1["df"]["reward_ind"].to_numpy(), 
                                tr2["df"]["reward_ind"].to_numpy()]
                    terminals = [tr0["df"]["terminal"].to_numpy(), 
                                tr1["df"]["terminal"].to_numpy(), 
                                tr2["df"]["terminal"].to_numpy()]
                    
                    path = utils.Path(obs, acs, rewards, next_obs, terminals)
                    paths.append(path)
                    step +=1
                    if step ==  num_transitions_to_sample:
                        return paths
        return paths
                    
            

    def train_agent(self, paths):
        all_logs = []
        idxs = np.random.permutation(np.arange(len(paths)))
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            path = paths[idxs[train_step]]
            # ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            
            ob = path['observation']
            ac = path['action']
            re = path['reward']
            next_ob = path['next_observation']
            terminal = path['terminal']

            # for i in range(self.params['train_batch_size']):
            for i in range(3):
                train_log = self.agent.train(ob[i], ac[i], re[i], next_ob[i], terminal[i])
                all_logs.append(train_log)
        return all_logs

    def eval_agent(self, paths):
        all_logs = {"Eval Mean KL Loss":[],
                        "Eval Mean AC Loss":[],
                        "Eval Mean Critic Loss":[]}
        log_means = {}

        for path in paths:
            ob = path['observation']
            ac = path['action']
            re = path['reward']
            next_ob = path['next_observation']
            terminal = path['terminal']
            # eval_log = self.agent.train(ob, ac, re, next_ob, terminal, eval=True)
            # print(len(path['action']))
            for i in range(len(path['action'])):
                eval_log = self.agent.eval(ob[i], ac[i], next_ob[i], re[i], terminal[i])
                # all_logs.append(eval_log)
                all_logs["Eval Mean KL Loss"].append(eval_log['Eval KL Loss'])
                all_logs["Eval Mean AC Loss"].append(eval_log['Eval AC Loss'])
                all_logs["Eval Mean Critic Loss"].append(eval_log['Eval value critic loss'])

        log_means  = {"Eval Mean KL Loss":np.mean(all_logs["Eval Mean KL Loss"]),
                        "Eval Mean AC Loss":np.mean(all_logs["Eval Mean AC Loss"]),
                        "Eval Mean Critic Loss":np.mean(all_logs["Eval Mean Critic Loss"])}
        return log_means


    ####################################
    ####################################
    
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        logs = OrderedDict()


        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        
        last_log = all_logs[-1]

        logs.update(last_log)
        


        print("\n~Running Evaluation~ \n")
        # Run eval
        # eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.eval_env, self.agent.eval_policy, self.params['eval_batch_size'], self.params['ep_len'])
        eval_paths = self.collect_training_trajectories(self.eval_indices, num_transitions_to_sample=10)
        logs.update(self.eval_agent(eval_paths))

        # logs['Buffer size'] = self.agent.replay_buffer.num_in_buffer

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()
