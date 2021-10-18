import numpy as np
import os
import math

from ddqn import Agent
from utils import plot_learning_curve
from update_design import HFSS

from neural_network import NN


def form_parameters(num_mclin, initial_values):
    half = int(math.ceil(num_mclin / 2))
    rest = int(num_mclin - half)
    values = initial_values[0]
    total = len(values)
    divides = int(total / 2)

    parameters = {}
    cnt = 1
    for i in range(half):
        key_name = 'MCLIN' + str(cnt)
        mclin_dict = {
            'W': values[i],
            'S': values[divides + i]
        }
        cnt += 1
        parameters[key_name] = mclin_dict

    for i in range(rest - 1, -1, -1):
        key_name = 'MCLIN' + str(cnt)
        mclin_dict = {
            'W': values[i],
            'S': values[divides + i]
        }
        cnt += 1
        parameters[key_name] = mclin_dict

    return parameters


if __name__ == '__main__':

    fps_path = os.path.join(os.getcwd(), r'Test\fpx\ParallelCoupledline_2.fpx')
    snp_path = os.path.join(os.getcwd(), r'Test\s2p\ParallelCoupledline_2.s2p')
    dpath = r"datasets/mclin.xlsx"

    sample = {
        'fo': 4.7e9,
        'bandwidth': 2.8e8,
        'length': 0.0117,
        'mclin': 5
    }
    nn = NN(dpath, sample)
    initial_guess = nn.predict_nn()

    params = form_parameters(sample['mclin'], initial_guess)
    half_cntr = int(math.ceil(sample['mclin'] / 2))
    rest_cntr = sample['mclin'] - half_cntr
    keys = list(params.keys())

    # Predicted From First Model
    for num in range(half_cntr):
        mclin_ids = [keys[num], keys[len(keys) - 1 - num]]
        mclin_params = params[mclin_ids[0]]
        mclin_w_range = (mclin_params['W'] * 0.9, mclin_params['W'] * 1.1)
        mclin_s_range = (mclin_params['S'] * 0.9, mclin_params['S'] * 1.1)
        tuning_parameter = list(mclin_params.keys())

        for tune_para in tuning_parameter:
            env = HFSS(fpx_path=fps_path, snp_path=snp_path, mclin_w_range=mclin_w_range, mclin_s_range=mclin_s_range,
                       fo=sample['fo'], bandwidth=sample['bandwidth'], length=sample['length'], mclin_ids=mclin_ids,
                       initial_guess=params, tuning_parameter=tune_para, return_loss=-10, insertion_loss=-50)

            num_action = len(env.action)
            num_state = len(env.state)
            print(f"*****  Initializing Fine-Tuning on {tune_para} Parameter for {mclin_ids}  *****".format(tune_para=tune_para, mclin_ids=str(mclin_ids)))
            print(f"no. of state: {num_state} | no. of action: {num_action}".format(num_state=num_state, num_action=num_action))

            n_episodes = 100
            load_checkpoint = False

            agent = Agent(gamma=0.99, epsilon=1.0, eps_min=0.01, eps_dec=1e-3, alpha=5e-4, input_dims=num_state,
                          n_actions=num_action, mem_size=1000000, batch_size=64, tune_parameter=tune_para, mclin_ids=str(mclin_ids))

            if load_checkpoint:
                agent.load_model()

            plot_loss_graph = 'Tuning-loss.png'
            if tune_para == "W":
                plot_loss_graph = os.path.join(os.getcwd(), r'ddqn/loss analysis/' + str(mclin_ids) + '_Width_Tuning-loss.png')
            if tune_para == "S":
                plot_loss_graph = os.path.join(os.getcwd(), r'ddqn/loss analysis/' + str(mclin_ids) + '_Space_Tuning-loss.png')
            scores, eps_history = [], []

            for ep in range(n_episodes):
                done = False
                score = 0
                observation = env.environment_reset()

                while not done:
                    action = agent.choose_action(observation)
                    observation_, reward, done = env.step(action)
                    score += reward
                    agent.remember(observation, action, reward, observation_, int(done))
                    agent.learn()
                    observation = observation_

                scores.append(score)
                avg_score = np.mean(scores[-100:])
                epsilon = agent.epsilon
                eps_history.append(epsilon)
                print(f'episode {ep} | score {score} | average score {avg_score} | epsilon {epsilon}'.format(ep=ep, score=score, avg_score=avg_score, epsilon=epsilon))

                if ep % 10 == 0 and ep > 0:
                    agent.save_model()
                if ep == n_episodes - 1:
                    agent.save_model()

            x = [i+1 for i in range(n_episodes)]
            plot_learning_curve(x, scores, eps_history, plot_loss_graph)
            params[mclin_ids[0]][tune_para] = env.value_update
            params[mclin_ids[1]][tune_para] = env.value_update
            print(f"*****  {mclin_ids} - {tune_para} Estimation Model Training completed *****".format(mclin_ids=str(mclin_ids), tune_para=tune_para))
            print("")
    print(params)