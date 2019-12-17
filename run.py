from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median, stdev
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os.path

# Run dqn with Tetris
def dqn():
    trainingAgent = False
    trainingHater = False
    env = Tetris(trainingAgent or trainingHater)
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 200 if (trainingAgent or trainingHater) else 10
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']
    agent_save_filepath = "keras_saved_maxbump.h5"
    # hater_save_filepath = "hater_changed_reward.h5"
    hater_save_filepath = "hater_best.h5"

    # Avg 135 || reward function = 1 + (lines_cleared ** 2)*self.BOARD_WIDTH - (.1)*self._bumpiness(self.board)[0]/self.BOARD_WIDTH
    # 200 death penalty
    # agent_save_filepath = "keras_saved_maxbump.h5"

    # Avg 25 || reward function = 1 + (lines_cleared ** 2)*self.BOARD_WIDTH
    # 2 death penalty
    # agent_save_filepath = "keras_saved.h5"

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size,
                     training=trainingAgent, agent_save_filepath=agent_save_filepath)

    hateris = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size,
                     training=trainingHater, agent_save_filepath=hater_save_filepath)
    env.hater = hateris

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            if len(current_state) == env.get_state_size() - 1 and trainingAgent:
                toBeAdded = current_state+[env.next_piece]
            elif len(current_state) == env.get_state_size() - 1 and trainingHater:
                toBeAdded = current_state+[env.current_piece]
            else: toBeAdded = current_state
            if trainingAgent:
                agent.add_to_memory(toBeAdded, next_states[best_action], reward, done)
            if trainingHater:
                hateris.add_to_memory(toBeAdded, next_states[best_action], -reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0 and trainingAgent:
            agent.train(batch_size=batch_size, epochs=epochs)
        if episode % train_every == 0 and trainingHater:
            hateris.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            std_score = stdev(scores[-log_every:])
            print(str(episode) + " Avg: " + str(avg_score) + "   Min: " + str(min_score) + "   Max: " + str(max_score) + "   Std: " + str(round(std_score, 2)))

        if episode == epsilon_stop_episode and trainingAgent:
            agent.save_agent("agent_stopEps.h5")
        if episode == epsilon_stop_episode and trainingHater:
            hateris.save_agent("hater_stopEps.h5")

    if trainingAgent: agent.save_agent("real_agent.h5")
    if trainingHater: hateris.save_agent("real_hater.h5")
    plt.plot(scores)
    plt.show()
if __name__ == "__main__":
    dqn()
