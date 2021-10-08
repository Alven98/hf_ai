import numpy as np
import time

from ddqn import Agent
from update_design import HFSS


if __name__ == '__main__':

    fps_path = r'Test\fpx\ParallelCoupledline_2.fpx'
    snp_path = r'Test\s2p\ParallelCoupledline_2.s2p'

    num_mclin = 5
    tuning_parameter = ["W", "S"]
    mclin_w_min = 2e-3
    mclin_w_max = 3e-3
    mclin_s_min = 0.2e-3
    mclin_s_max = 0.3e-3
    cf = 0.5
    el = 90

    start_time = time.time()

    # Predicted From First Model (TODO: Predict W, S, CF, EL from First Model CNN)
    for num in range(num_mclin):

        mclin_id = num + 1
        if num in [0, 4]:
            mclin_s_min = 0.2e-3
            mclin_s_max = 0.3e-3
            cf = 0.5894
            el = 90.6
        elif num in [1, 3]:
            mclin_s_min = 2e-3
            mclin_s_max = 3e-3
            cf = 0.876992
            el = 90.69
        else:
            mclin_s_min = 2e-3
            mclin_s_max = 3e-3
            cf = 0.901332
            el = 90.69

        mclin_w_range = (mclin_w_min, mclin_w_max)
        mclin_s_range = (mclin_s_min, mclin_s_max)

        for tune_para in tuning_parameter:
            env = HFSS(fps_path, snp_path, mclin_w_range, mclin_s_range, cf, el, tune_para, mclin_id)
            num_action = len(env.action)
            num_state = len(env.state)
            print(f"*****  Initializing Fine-Tuning on {tune_para} Parameter for MCLIN{mclin_id}  *****".format(tune_para=tune_para, mclin_id=mclin_id))
            print(f"no. of state: {num_state} | no. of action: {num_action}".format(num_state=num_state, num_action=num_action))

            n_episodes = 1

            # Load model
            agent = Agent(gamma=0.99, epsilon=0.01, eps_min=0.01, eps_dec=1e-3, alpha=5e-4, input_dims=num_state,
                          n_actions=num_action, mem_size=1000000, batch_size=64, tune_parameter=tune_para, mclin_id=mclin_id)
            agent.load_model()

            scores, eps_history = [], []

            for ep in range(n_episodes):
                done = False
                score = 0
                observation = env.environment_reset()

                while not done:
                    action = agent.choose_action(observation)
                    observation_, reward, done = env.step(action)
                    score += reward

                print(f"Predicted MCLIN{mclin_id} {tune_para} : ", env.params)

                scores.append(score)
                avg_score = np.mean(scores[-100:])
                epsilon = agent.epsilon
                eps_history.append(epsilon)
                print(f'episode {ep} | score {score} | average score {avg_score} | epsilon {epsilon}'.format(ep=ep, score=score, avg_score=avg_score, epsilon=epsilon))

            print(f"*****  MCLIN{mclin_id} - {tune_para} Estimation Model Testing completed *****".format(mclin_id=mclin_id, tune_para=tune_para))
            print('')

    finish_time = time.time()
    elapsed_time = finish_time - start_time
    print(f"Time Elapsed : {elapsed_time} sec".format(elapsed_time=elapsed_time))
