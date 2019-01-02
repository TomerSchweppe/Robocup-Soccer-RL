#!/home/deep7/robocup_ofir_tomer/tensorflow/bin/python
"""
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import gym_soccer
import time
import sys
from replay_buffer import ReplayBuffer
import os
from shutil import copyfile

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
MAX_EPISODES_TEST = 100
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
EPSILON_START = 1.
EPSILON_END = 0.1

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True

# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'SoccerAgainstStatic-v0'
# base output directory
BASE_DIR = './results/' + time.strftime("%Y_%m_%d_%H_%M")
# Directory for storing gym results
MONITOR_DIR = BASE_DIR + '/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = BASE_DIR + '/tf_ddpg'
RANDOM_SEED = None
# Size of replay buffer
BUFFER_SIZE = 500000
MINIBATCH_SIZE = 32
# Debug Telemetry
LOAD_NETWORK_FROM_FILE = False
# ===========================
#   Soccer Parameters
# ===========================
SOCCER_ACTION_DIM = 8
SOCCER_ACTION_BOUND_SPACE = [1., 1., 1., 100., 360., 360., 100., 360.]
SOCCER_ACTION_BOUND_LOW = [0., 0., 0., 0., -180., -180., 0., -180.]
ACTION_LOOKUP = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
SKILLS_NUM = 2


# ===========================
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound_space, action_bound_low, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound_space = action_bound_space
        self.action_bound_low = action_bound_low
        self.learning_rate = tf.constant(learning_rate)
        self.tau = tau

        # Actor Network
        self.inputs, self.scaled_out_head1, self.scaled_out_head2 = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out_head1, self.target_scaled_out_head2 = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in xrange(len(self.target_network_params))]

        self.init_target_network_params = \
            [self.target_network_params[i].assign(self.network_params[i])
             for i in xrange(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients_head1 = tf.gradients(
            self.scaled_out_head1, self.network_params, -self.action_gradient)

        # grads_head1, _ = tf.clip_by_global_norm(self.actor_gradients_head1, 10000)
        grads_head1 = self.actor_gradients_head1

        self.actor_gradients_head2 = tf.gradients(
            self.scaled_out_head2, self.network_params, -self.action_gradient)

        # grads_head2, _ = tf.clip_by_global_norm(self.actor_gradients_head2, 10000)
        grads_head2 = self.actor_gradients_head2

        # Learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
        #                                                 30000 * 180, 0.1, staircase=True)
        start_decay_step = int(MAX_EPISODES * 160 / 2)
        decay_steps = int(start_decay_step / 5)
        decay_factor = 0.5
        self.learning_rate = tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True))
        tf.summary.scalar('Actor Learning Rate', self.learning_rate)

        # Optimization Op
        self.optimize1 = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(grads_head1, self.network_params), global_step=self.global_step)
        self.optimize2 = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(grads_head2, self.network_params), global_step=self.global_step)

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.activations.leaky_relu(tflearn.fully_connected(inputs, 400), alpha=0.01)
        net = tflearn.activations.leaky_relu(tflearn.fully_connected(net, 300), alpha=0.01)
        net = tflearn.activations.leaky_relu(tflearn.fully_connected(net, 200), alpha=0.01)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init1 = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        w_init2 = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        head1 = tflearn.fully_connected(net, self.a_dim, activation='sigmoid', weights_init=w_init1)
        head2 = tflearn.fully_connected(net, self.a_dim, activation='sigmoid', weights_init=w_init2)
        # Scale output to -action_bound to action_bound

        scaled_out_head1 = tf.add(tf.multiply(head1, self.action_bound_space), self.action_bound_low)
        scaled_out_head2 = tf.add(tf.multiply(head2, self.action_bound_space), self.action_bound_low)
        # scaled_out = tf.add(tf.multiply(tf.multiply(tf.multiply(out, 1 - out), 4), self.action_bound_space),self.action_bound_low)
        return inputs, scaled_out_head1, scaled_out_head2

    def train(self, inputs, a_gradient):
        if inputs[0][47] < 0:
            return self.sess.run([self.optimize1, ], feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            })
        return self.sess.run([self.optimize2, ], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        if inputs[0][47] < 0:
            return self.sess.run(self.scaled_out_head1, feed_dict={
                self.inputs: inputs
            })
        return self.sess.run(self.scaled_out_head2, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        if inputs[0][47] < 0:
            return self.sess.run(self.target_out_head1, feed_dict={
                self.target_inputs: inputs
            })
        return self.sess.run(self.target_scaled_out_head2, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def init_target_network(self):
        self.sess.run(self.init_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        input_net = tflearn.activations.leaky_relu(tflearn.fully_connected(inputs, 400), alpha=0.01)
        action_net = tflearn.activations.leaky_relu(tflearn.fully_connected(action, 100), alpha=0.01)
        # Add the action tensor in the 2nd hidden layer
        net = tflearn.activations.leaky_relu(tflearn.fully_connected(tf.concat((input_net, action_net), 1), 300),
                                             alpha=0.01)
        net = tflearn.activations.leaky_relu(tflearn.fully_connected(net, 200), alpha=0.01)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.optimize, self.out, self.loss], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = tf.constant(learning_rate)
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i],
                                                                            1. - self.tau))
                for i in xrange(len(self.target_network_params))]

        self.init_target_network_params = \
            [self.target_network_params[i].assign(self.network_params[i])
             for i in xrange(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
        #                                                 30000 * 180, 0.1, staircase=True)
        start_decay_step = int(MAX_EPISODES * 160 / 2)
        decay_steps = int(start_decay_step / 5)
        decay_factor = 0.5
        self.learning_rate = tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True))
        tf.summary.scalar('Critic Learning Rate', self.learning_rate)

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        grads = tf.gradients(self.loss, self.network_params)
        # grads, _ = tf.clip_by_global_norm(grads, 10000)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = optimizer.apply_gradients(zip(grads, self.network_params), global_step=self.global_step)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        # Get the norm of critic gradients for telemetry
        self.grads_norm = tf.global_norm(self.action_grads)

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run([self.action_grads, self.grads_norm], feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def init_target_network(self):
        self.sess.run(self.init_target_network_params)


# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries(actor, critic):
    # tflearn.summaries.add_trainable_vars_summary(actor.network_params, name_prefix='actor', name_suffix='params')
    # tflearn.summaries.add_trainable_vars_summary(critic.network_params, name_prefix='critic', name_suffix='params')
    # tflearn.summaries.add_trainable_vars_summary(actor.target_network_params, name_prefix='target actor', name_suffix='params')
    # tflearn.summaries.add_trainable_vars_summary(critic.target_network_params, name_prefix='target critic', name_suffix='params')
    episode_loss = tf.Variable(0., trainable=False)
    tf.summary.scalar("Loss", episode_loss)
    episode_exe_time = tf.Variable(0., trainable=False)
    tf.summary.scalar("Episode Execute Time", episode_exe_time)
    grads_norm = tf.Variable(0., trainable=False)
    tf.summary.scalar("Gradients Norm", grads_norm)
    episode_reward = tf.Variable(0., trainable=False)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0., trainable=False)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    num_of_steps = tf.Variable(0., trainable=False)
    tf.summary.scalar("Number of steps per episode", num_of_steps)

    summary_vars = [episode_reward, episode_ave_max_q, episode_loss, grads_norm, episode_exe_time, num_of_steps]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def step_array(pred):
    arr = pred[0]
    return tuple(np.concatenate((np.array([np.argmax(arr[0:2 + 1])]), arr[3:])))


# ===========================
#   Agent Training
# ===========================


def train(sess, saver, env, actor, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries(actor, critic)

    sess.run(tf.global_variables_initializer())

    if LOAD_NETWORK_FROM_FILE:
        saver.restore(sess, "/tmp/model.ckpt")

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    copyfile(os.path.realpath(__file__), BASE_DIR + "/code.py")

    # Initialize target network weights
    actor.init_target_network()
    critic.init_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    eps = EPSILON_START

    step_counter = 0

    for i in xrange(MAX_EPISODES):

        s = env.reset()
        while s[47] > 0:
            s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        ep_ave_max_loss = 0
        grads_norm = 0

        # Start time measure for episode
        start_time = time.time()

        for j in xrange(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            if (np.random.random() < eps):
                action = np.asarray(env.action_space.sample())
                a = np.concatenate((np.asarray(ACTION_LOOKUP[int(action[0])]), action[1:]))
            else:
                a = actor.predict(np.reshape(s, (1, actor.s_dim)))
                action = step_array(a)

            s2, r, terminal, info = env.step(action)
            r *= 1000

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE ** 2:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                _, predicted_q_value, tmp_loss = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)
                ep_ave_max_loss += np.amax(tmp_loss)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads, norm = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                grads_norm += norm

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                if eps > EPSILON_END:
                    eps -= 0.00009

            s = s2
            ep_reward += r

            if terminal:

                step_counter += j + 1

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: ep_ave_max_loss / float(j),
                    summary_vars[3]: grads_norm / float(j),
                    summary_vars[4]: time.time() - start_time,
                    summary_vars[5]: step_counter
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print '| Reward: %f' % ep_reward, " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j)), '| Epsilon: %.5f' % (eps)

                if (i + 1) % 1000 == 0:
                    save_path = saver.save(sess, SUMMARY_DIR + "/model.ckpt", global_step=(i + 1))
                    print("Model saved in file: %s" % save_path)
                break


def test(sess, saver, env, actor, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries(actor, critic)

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, sys.argv[1])

    # writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    for i in xrange(MAX_EPISODES_TEST):

        s = env.reset()
        while s[47] > 0:
            s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        ep_ave_max_loss = 0
        grads_norm = 0

        # Start time measure for episode
        start_time = time.time()

        for j in xrange(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            time.sleep(0.01)
            action = step_array(a)

            s2, r, terminal, info = env.step(action)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples

            s = s2
            ep_reward += r

            if terminal:
                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: ep_reward,
                #     summary_vars[1]: ep_ave_max_q / float(j),
                #     summary_vars[2]: ep_ave_max_loss / float(j),
                #     summary_vars[3]: grads_norm / float(j),
                #     summary_vars[4]: time.time() - start_time
                # })
                #
                # writer.add_summary(summary_str, i)
                # writer.flush()

                print '| Reward: %f' % ep_reward, " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j))

                break


def main(_):
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = SOCCER_ACTION_DIM
        action_bound_space = SOCCER_ACTION_BOUND_SPACE
        action_bound_low = SOCCER_ACTION_BOUND_LOW

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound_space, action_bound_low,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        saver = tf.train.Saver(max_to_keep=0)

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        if len(sys.argv) > 1:
            test(sess, saver, env, actor, critic)
        else:
            train(sess, saver, env, actor, critic)

        # if GYM_MONITOR_EN:
        #     env.monitor.close()


if __name__ == '__main__':
    tf.app.run()
