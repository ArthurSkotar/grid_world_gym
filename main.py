from typing import Union

import gym
import numpy as np
from gym.core import ObsType

import gym_examples
import random
from itertools import permutations
import itertools as it

col = list(it.product('01234', repeat=4))


def get_state(item: dict):
    agent = item['agent']
    target = item['target']
    agent_target = np.concatenate((agent, target), axis=0)
    mapped = tuple(map(str, agent_target))
    print(mapped)
    ind = col.index(mapped)
    print(ind)
    return ind


def train(env, num_episodes, max_steps):
    # global env, qtable, num_episodes, max_steps, state, done, s, action, new_state, reward, info
    state_size = 625  # total number of states (S)
    action_size = env.action_space.n  # total number of actions (A)
    # initialize a qtable with 0's for all Q-values
    qt = np.zeros((state_size, action_size))
    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        status = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qt[get_state(state), :])

            # take action and observe reward
            new_state, reward, status, info = env.step(action)

            # Q-learning algorithm
            qt[get_state(state), action] = qt[get_state(state), action] + learning_rate * (
                    reward + discount_rate * np.max(qt[get_state(new_state), :]) - qt[get_state(state), action])

            # Update to our new state
            state = new_state
            # if done, finish episode
            if status == True:
                break
        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)
    return qt


if __name__ == '__main__':
    env = gym.make('gym_examples/GridWorld-v0', render_mode="human")
    # training variables
    num_episodes = 1000
    max_steps = 99  # per episode
    # Uncomment this to load already trained network
    # with open('test.npy', 'rb') as f:
    #     qtable = np.load(f)
    qtable = train(env, num_episodes, max_steps)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[get_state(state), :])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

    env.close()
    #     observation, info = env.reset(seed=100, return_info=True)
    #     env.render()
    #     action = env.action_space.sample()  # User-defined policy function
    #     observation, reward, done, info = env.step(action)
    #     if done:
    #         observation, info = env.reset(return_info=True)
    # env.close()
