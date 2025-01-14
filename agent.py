
import datetime
import keyboard
import tensorflow as tf

from tf_agents.agents.dqn.dqn_agent import D3qnAgent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks.dueling_q_network import DuelingQNetwork
from tf_agents.environments import parallel_py_environment
from tf_agents.system import multiprocessing

# from PERBuffer import TFPrioritizedReplayBuffer
from environment import TetrisEnvironment

import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'


num_iterations = 500_000#6_000_000

initial_collect_steps = 128  
collect_steps_per_iteration = 10
replay_buffer_max_length = 220_000  

batch_size = 268
learning_rate = 4.5e-4
log_interval = 200

num_eval_episodes = 50  
eval_interval = 1000

end_early = False
    

# Create Agent 
def Agent(train_env):
    train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate,
            train_step_counter,   
            int(0.15*num_iterations),
            0.37,
        )
    )
    
    # Set agent parameters
    agent = D3qnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=DuelingQNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec(),
            activation_fn=tf.keras.activations.relu,
            q_layer_activation_fn=tf.keras.activations.linear,
            fc_layer_params=(200, 200, 200)),
        optimizer=optimizer,
        gamma=0.94,
        epsilon_greedy=tf.compat.v1.train.polynomial_decay(
            learning_rate=1.0,
            global_step=train_step_counter,
            decay_steps=int(0.7*num_iterations),
            power=2.7,  
            end_learning_rate=0.04),
        target_update_period=100,   
        target_update_tau=0.25,
        td_errors_loss_fn=common.element_wise_huber_loss,
        train_step_counter=train_step_counter,
        debug_summaries=True,
        summarize_grads_and_vars=False
        )
    
    agent.initialize()

    return agent


# Train a new model or run already trained model
def main(argv):
    if(len(argv) > 1):
        tf_env = tf_py_environment.TFPyEnvironment(TetrisEnvironment(True))
        saved_policy = tf.compat.v2.saved_model.load(argv[1])
        for _ in range(int(argv[2]) if len(argv) > 2 else 10):
            policy_state = saved_policy.get_initial_state(batch_size=3)
            time_step = tf_env.reset()  
            while not time_step.is_last():
                action_step = saved_policy.action(time_step, policy_state)
                policy_state = action_step.state
                time_step = tf_env.step(action_step.action)

    else:
        train_agent()
    

def train_agent():
    
    # Setup parallel data collection environments
    train_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment([TetrisEnvironment]*20))
    eval_env = tf_py_environment.TFPyEnvironment(TetrisEnvironment())
    
    agent = Agent(train_env)
    
    collect_policy = agent.collect_policy

    # Create experience replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,      
        batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
    
    # setup the buffer observer
    replay_observer = [replay_buffer.add_batch]
    
    # Driver to collect data from envrionment
    driver = dynamic_step_driver.DynamicStepDriver(
        train_env, collect_policy, replay_observer, num_steps=collect_steps_per_iteration)
    
   
    
    # Create the dataset from buffer
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=8,
        sample_batch_size=batch_size,
        num_steps=2,
    single_deterministic_pass=False).prefetch(8)
    
    iterator = iter(dataset)
    
    
    # Setup certain dynamic variables
    beta =  tf.compat.v1.train.polynomial_decay(
            learning_rate=0.0001,
            global_step=agent.train_step_counter,
            decay_steps=int(0.8*num_iterations),
            power=4,  
            end_learning_rate=1.0)
    
    epsilon = tf.compat.v1.train.polynomial_decay(
            learning_rate=1.0,
            global_step=agent.train_step_counter,
            decay_steps=int(0.7*num_iterations),
            power=2.7,  
            end_learning_rate=0.04)
    
    learn_rate = tf.compat.v1.train.exponential_decay(
            learning_rate,
            agent.train_step_counter,   
            int(0.15*num_iterations),
            0.37,
        )

    agent.train = common.function(agent.train)

    agent.train_step_counter.assign(0)
    
    time_step = train_env.reset()
    
    current_time = datetime.datetime.now()
    
    train_dir = os.path.join("./train/logs", "{0}-{1}-{2}".format(current_time.day, current_time.hour, current_time.minute))    
    train_summary_writer = tf.summary.create_file_writer(
                train_dir, flush_millis=10000, name="test")
    train_summary_writer.set_as_default()
    
    for i in range(100):
        driver.run()
    
    #tf.summary.trace_on(graph=True)
        
    # Model policy saver
    policy_dir = os.path.join("./train/", 'models')
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)    
    
    max_return = 80     
    
    # Main training loop
    for i in range(num_iterations):
        if(end_early):
            return
        
        
        time_step, _ = driver.run(time_step)

        experience, buffer_info = next(iterator)
        learning_weights = (1/(tf.clip_by_value(buffer_info.probabilities, 0.000001, 1.0)*batch_size))**beta()
        train_loss, _ = agent.train(experience=experience); """Not sure if good:  weights=learning_weights"""
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1:.6f}: beta: {2:.6f}: epsilon: {3:.6f}: learning rate: {4:.6f}'.format(step, train_loss, beta(), epsilon(), learn_rate()))
            tf.compat.v2.summary.scalar(name="epsilon", data=epsilon(), step=step)
            tf.compat.v2.summary.scalar(name="learning rate", data=learn_rate(), step=step)
            
            
        if step % eval_interval == 0:
            avg_return, avg_length = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}: Average Length = {2}'.format(step, avg_return, avg_length))  
            
            tf.compat.v2.summary.scalar(name="avg. return", data=avg_return, step=step)
            #tf.compat.v2.summary.trace_export(name="DuelingQNetwork", step=step)
            
            
            if(avg_return > max_return):
                max_return = avg_return
                current_time = datetime.datetime.now()
                tf_policy_saver.save(os.path.join(policy_dir, "{0:.3f}-{1}-{2}-{3}-{4}.v22".format(avg_return, current_time.day, current_time.hour, current_time.minute, step)))
    
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
  return avg_return.numpy()[0], avg_length


# tf.debugging.experimental.enable_dump_debug_info(os.path.join("./train", ''), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
if __name__ == '__main__':
    multiprocessing.handle_main(main)
