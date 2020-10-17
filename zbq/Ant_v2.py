import gym
env = gym.make('Ant-v2')
print('env.action_space: ', env.action_space)
print('env.action_space.sample(): ', env.action_space.sample())
print('env.observation_space: ', env.observation_space)
print('env.observation_space.sample(): ', env.observation_space.sample())

for i_episode in range(50):
    observation = env.reset() # get original feedback
    for t in range(100):
        # env.render() # render
        # print(observation)
        action = env.action_space.sample() # produce a random action 
        # key code:
        observation, reward, done, info = env.step(action) # get feedback of the action
        print(observation, reward, done, info)
        if done:
            print('Episode {} finished after {} timesteps'.format(i_episode, t+1))
            break
env.close()



# from gym import envs
# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)

# import gym

# env = gym.make('Ant-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     # take a random action
#     env.step(env.action_space.sample())