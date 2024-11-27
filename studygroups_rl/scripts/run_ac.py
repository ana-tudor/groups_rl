import os
import time

from studygroups_rl.agents.ac_agent import AWACAgent
from studygroups_rl.infrastructure.rl_trainer import RL_Trainer


class AC_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################
        self.params = params

        train_args = {
            'num_agent_init_steps': params['num_agent_init_steps'],
            'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'target_update_freq': params['target_update_freq'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'use_boltzmann': params['use_boltzmann'],
            'n_iter':params['n_iter']
        }

        agent_args={
            'total_size':params['total_size'],
            'input_dim':params['input_dim'],
            'hidden_dim_ae':params['hidden_dim_ae'],
            'n_hidden_layers_ae':params['n_hidden_layers_ae'],
            'latent_dim':params['latent_dim'],
            'layer_activation_ae':params['layer_activation_ae'],
            'output_activation':params['output_activation'],
            'use_dropout':params['use_dropout'],
            'k':params['total_size']//params['group_size'],
            'group_size':params['group_size'],
            'lr':params['learning_rate'],
            'ae_lr':params['ae_lr'],
            'dropout_rate':params['dropout_rate']
            }

        critic_args = {
            'n_critic_layers': params['n_critic_layers'],
            'critic_size': params['critic_size'],
            'critic_lr': params['critic_lr'],
            'num_target_updates': params['num_target_updates'],
            'num_grad_steps_per_target_update': params['num_grad_steps_per_target_update'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
        }

        # train_args = {
        #     'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        #     'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
        #     'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
        # }

        agent_params = {**agent_args, **critic_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = AWACAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


def main():

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', type=str, default='CartPole-v0')
    # parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)
    parser.add_argument('--total_size', '-ts', type=int, default=500)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=10)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=20)

    parser.add_argument('--batch_size', '-b', type=int, default=20) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=3) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=20) ##steps used per gradient step

    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--target_update_freq', '-tuf', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=2)
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    # parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=10)

    #Train args
    parser.add_argument('--num_agent_init_steps', type=int, default=60000)
    parser.add_argument('--use_boltzmann', action='store_true')
    parser.add_argument('--offline_exploitation', action='store_true')

    #Agent args
    parser.add_argument('--input_dim', type=int, default=134)
    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--hidden_dim_ae', type=int, default=24)
    parser.add_argument('--n_hidden_layers_ae', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--layer_activation_ae', type=str, default='relu')
    parser.add_argument('--output_activation', type=str, default='identity')
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--dropout_rate', '-drop_rate', type=float, default=0.0)
    parser.add_argument('--dec_lr',  type=float, default=1e-4)
    parser.add_argument('--ae_lr',  type=float, default=1e-5)
    
    #Critic args
    parser.add_argument('--n_critic_layers', type=int, default=2)
    parser.add_argument('--critic_size', type=int, default=128)
    parser.add_argument('--critic_lr',  type=float, default=0.01)


    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    # for policy gradient, we made a design decision
    # to force batch_size = train_batch_size
    # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
    params['batch_size'] = 1
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './rl_data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_studygroups_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = AC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
