import numpy as np

from dueling_ddqn import Agent
from utils import plot_learning_curve
from update_design import HFSS


if __name__ == '__main__':

    fps_path = r'Test\fpx\ParallelCoupledline.fpx'
    snp_path = r'Test\s2p\ParallelCoupledline_2.s2p'

    # Adjustables
    num_ports = 2
    f_min = 3e9
    f_max = 4e9
    f_mid = 3.5e9
    mclin_w_min = 2e-3
    mclin_w_max = 3e-3
    mclin_s_min = 10e-6
    mclin_s_max = 3000e-6
    passband_return_loss = -10
    stopband_insertion_loss = -50

    fc1 = f_mid - 0.1e9
    fc2 = f_mid + 0.1e9
    mclin_w_range = [mclin_w_min, mclin_w_max]
    mclin_s_range = [mclin_s_min, mclin_s_max]

    env = HFSS(fps_path, snp_path, f_min, f_max, f_mid, passband_return_loss, stopband_insertion_loss,
               num_ports, mclin_w_range, mclin_s_range)

    num_action = len(env.action)
    num_state = len(env.state)

    print("no. of state: ", num_state, " | no. of action: ", num_action)

    n_episodes = 800000
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, eps_min=0.01, eps_dec=1e-3, lr=5e-4, input_dims=[10],
                  n_actions=num_action, mem_size=1000000, batch_size=64, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'tuning-DuelingDDQN-loss.png'
    scores, eps_history = [], []

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.environment_reset()
        # print(observation)

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            print(reward)
            print(observation_)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        eps_history.append(agent.epsilon)
        print('episode ', i, ' | score %.2f' % score,
              ' | average score %.2f' % avg_score, ' | epsilon %.2f' % agent.epsilon)

        if i % 10 == 0 and i > 0:
            agent.save_models()
        if i == n_episodes - 1:
            agent.save_models()

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, scores, eps_history, filename)
