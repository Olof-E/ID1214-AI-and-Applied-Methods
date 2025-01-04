
import tensorflow as tf

from tf_agents.agents.dqn.dqn_agent import D3qnAgent
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.networks.dueling_q_network import DuelingQNetwork
from tf_agents.environments import parallel_py_environment
from tf_agents.system import multiprocessing

# from PERBuffer import TFPrioritizedReplayBuffer
from PERBuffer import TFPrioritizedReplayBuffer
from environment import TetrisEnvironment

import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


num_iterations = 250_000 # @param {type:"integer"}

initial_collect_steps = 128  # @param {type:"integer"}
collect_steps_per_iteration = 16*2 # @param {type:"integer"}
replay_buffer_max_length = 25_000  # @param {type:"integer"}

batch_size = 128      # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 50  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
    


def Agent(train_env):
    train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate,
            train_step_counter,
            int(0.33*num_iterations),
            0.8,
        )
    )

      
    agent = D3qnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=DuelingQNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec(),
            activation_fn=tf.keras.activations.relu,
            q_layer_activation_fn=tf.keras.activations.linear,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'),
            fc_layer_params=(64, 64, 64)),
        optimizer=optimizer,
        gamma=0.99,
        epsilon_greedy=tf.compat.v1.train.polynomial_decay(
            learning_rate=1.0,
            global_step=train_step_counter,
            decay_steps=int(0.75*num_iterations),
            power=6,  
            end_learning_rate=0.001),
        target_update_period=250,   
        target_update_tau=0.1,
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


def TetrisEnvironment1():
    return TetrisEnvironment()


def train_agent(argv):
    
    train_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment([TetrisEnvironment]*16))
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
    
    train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=collect_steps_per_iteration),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=collect_steps_per_iteration),
    ]

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env, collect_policy, replay_observer, num_steps=collect_steps_per_iteration)
    
    driver.run()
    
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2,
    single_deterministic_pass=False).prefetch(3)
    
    iterator = iter(dataset)
    
    episode_len = []

    beta =  tf.compat.v1.train.polynomial_decay(
            learning_rate=0.0001,
            global_step=agent.train_step_counter,
            decay_steps=int(0.8*num_iterations),
            power=4,  
            end_learning_rate=1.0)
    
    epsilon = tf.compat.v1.train.polynomial_decay(
            learning_rate=1.0,
            global_step=agent.train_step_counter,
            decay_steps=int(0.75*num_iterations),
            power=6,  
            end_learning_rate=0.001)
    
    learn_rate = tf.compat.v1.train.exponential_decay(
            learning_rate,
            agent.train_step_counter,   
            int(0.33*num_iterations),
            0.8,
        )
    #collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)
    
    # for i in range(initial_collect_steps):
    #     final_time_step, policy_state = driver.run()
    
    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    
    time_step = train_env.reset()
    
    
    # train_dir = os.path.join("./train", '')    
    # train_summary_writer = tf.summary.create_file_writer(
    #             train_dir, flush_millis=10000)
    # train_summary_writer.set_as_default()
    
    


    for i in range(num_iterations):

        
        time_step, _ = driver.run(time_step)

        experience, buffer_info = next(iterator)
        learning_weights = (1/(tf.clip_by_value(buffer_info.probabilities, 0.000001, 1.0)*batch_size))**beta()
        train_loss, extra = agent.train(experience=experience, weights=learning_weights)
        # replay_buffer.update_batch(buffer_info.ids, extra.td_loss) 
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:    
            print('step = {0}: loss = {1:.6f}: Average episode length: {2:.6f}: beta: {3:.6f}: epsilon: {4:.6f}: learning rate: {5:.6f}'.format(step, train_loss, train_metrics[3].result().numpy(), beta(), epsilon(), learn_rate()))
            # for train_metric in train_metrics:
            #     train_metric.tf_summaries(train_step=agent.train_step_counter, step_metrics=train_metrics[:2])

        if step % eval_interval == 0:
            avg_return, avg_length = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}: Average Length = {2}'.format(step, avg_return, avg_length))  
            print(agent.collect_policy)
            returns.append(avg_return)
            episode_len.append(avg_length)


    plt.plot(episode_len)
    plt.plot(returns)
    plt.show()
    
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  total_steps = 0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
      total_steps += 1
    total_return += episode_return

  avg_return = total_return / num_episodes
  avg_length = total_steps / num_episodes
  return avg_return, avg_length


# tf.debugging.experimental.enable_dump_debug_info(os.path.join("./train", ''), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

if __name__ == '__main__':
    multiprocessing.handle_main(train_agent)
