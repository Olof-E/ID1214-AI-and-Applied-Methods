
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver

from PERBuffer import TFPrioritizedReplayBuffer
from environment import TetrisEnvironment

import matplotlib.pyplot as plt

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'


num_iterations = 50000 # @param {type:"integer"}

initial_collect_steps = 500  # @param {type:"integer"}
collect_steps_per_iteration = 5 # @param {type:"integer"}
replay_buffer_max_length = 32000  # @param {type:"integer"}

batch_size = 64      # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}



def QNetwork(env):
    fc_layer_params = (128, 100)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.15))
    
    return sequential.Sequential(dense_layers + [q_values_layer])
    
# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))
    


def Agent(train_env):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=QNetwork(train_env),
        optimizer=optimizer,
        gamma=0.9,
        epsilon_greedy=0.15,
        target_update_period=100,
        target_update_tau=0.9,
        target_q_network=QNetwork(train_env),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    
    return agent

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  # print(traj)
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)



def train_agent():
    train_env = tf_py_environment.TFPyEnvironment(TetrisEnvironment())
    eval_env = tf_py_environment.TFPyEnvironment(TetrisEnvironment())
    agent = Agent(train_env)
    
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

    # replay_buffer = TFPrioritizedReplayBuffer(
    #     data_spec=agent.collect_data_spec,
    #     batch_size=train_env.batch_size,
    #     max_length=replay_buffer_max_length)

    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,      
        batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

    replay_observer = [replay_buffer.add_batch]
    
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2,
    single_deterministic_pass=False).prefetch(3)
    
    iterator = iter(dataset)
    
    
    train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
    ]


    
    
    #agent.train = common.function(agent.train)
    
    # Reset the train step.

    # Evaluate the agent's policy once before training.
    #avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    #returns = [avg_return]

    # Reset the environment.
    #time_step = train_env.reset()

    # driver = py_driver.PyDriver(
    #     train_env,
    #     py_tf_eager_policy.PyTFEagerPolicy(
    #     agent.collect_policy, use_tf_function=True),
    #     [replay_observer + train_metrics],
    #     max_steps=initial_collect_steps)

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env, collect_policy, replay_observer + train_metrics, num_steps=collect_steps_per_iteration)

    # driver = dynamic_step_driver.DynamicStepDriver(
    #     train_env,
    #     random_policy,
    #     observers=replay_observer + train_metrics,
    # num_steps=2)
    
    episode_len = []

    # for _ in range(num_iterations):

    #     # Collect a few steps and save to the replay buffer.
    #     time_step, _ = driver.run(time_step)

    #     # Sample a batch of data from the buffer and update the agent's network.
    #     experience, unused_info = next(iterator)
    #     train_loss = agent.train(experience).loss

    #     step = agent.train_step_counter.numpy()

    #     if step % log_interval == 0:
    #         print('step = {0}: loss = {1}'.format(step, train_loss))

    #     if step % eval_interval == 0:
    #         avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    #         print('step = {0}: Average Return = {1}'.format(step, avg_return))
    #         returns.append(avg_return)

    beta_PER_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.00,
    end_learning_rate=1.00,
    decay_steps = num_iterations)

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for i in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, buffer_info = next(iterator)
        learning_weights = (1/(tf.clip_by_value(buffer_info.probabilities, 0.000001, 1.0)*batch_size))**beta_PER_fn(i)
        train_loss, extra = agent.train(experience=experience, weights=learning_weights)
        # replay_buffer.update_batch(buffer_info.ids, extra.td_loss)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}: beta: {2}'.format(step, train_loss, beta_PER_fn(i)))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            print(agent.collect_policy)
            returns.append(avg_return)


    # final_time_step, policy_state = driver.run()

    # for i in range(num_iterations):
    #     final_time_step, _ = driver.run(final_time_step, policy_state)

    #     experience, _ = next(iterator)
    #     train_loss = agent.train(experience=experience)
    #     step = agent.train_step_counter.numpy()

    #     if step % log_interval == 0:
    #         print('Number of trajectories recorded by P1:', replay_buffer.num_frames().numpy())
    #         print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    #         episode_len.append(train_metrics[3].result().numpy())
    #         print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

    #     if step % eval_interval == 0:
    #         avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    #         print('step = {0}: Average Return = {1}'.format(step, avg_return))
    #         print(driver._policy)
    plt.plot(episode_len)
    plt.show()
    
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

train_agent()
