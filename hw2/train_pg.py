import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
from threading import Thread, Lock

#============================================================================================#
# Utilities
#============================================================================================#


def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None,
        reuse_hidden_layers=False,
        reuse_scope_name=None
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#
    if reuse_hidden_layers:
        with tf.variable_scope(reuse_scope_name, reuse=True):
            x = input_placeholder
            for l in range(n_layers):
                x = tf.layers.dense(x, size, activation=activation, reuse=True, name='dense{}'.format(l))

        with tf.variable_scope(scope):
            return tf.layers.dense(x, output_size, activation=output_activation)
    else:
        with tf.variable_scope(scope):
            x = input_placeholder
            for l in range(n_layers):
                x = tf.layers.dense(x, size, activation=activation, reuse=None, name='dense{}'.format(l))

            return tf.layers.dense(x, output_size, activation=output_activation)
        

def pathlength(path):
    return len(path["reward"])


def normalize(array, epsilon=1e-7):
    return (array - array.mean(axis=0, keepdims=True))/(array.std(axis=0, keepdims=True) + epsilon)


def rescale(array, mean=0, std=1):
    return array * std + mean


class TrajectionRunner(Thread):

    def __init__(self, sess: tf.Session, sy_sampled_ac, sy_ob_no, env_name, max_path_length, min_timesteps_per_batch):
        self.paths = []
        self.session = sess
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.env_name = env_name
        self.sy_sampled_ac = sy_sampled_ac
        self.sy_ob_no = sy_ob_no
        self.total_timesteps = 0
        super(TrajectionRunner, self).__init__()

    def run(self):

        env = gym.make(self.env_name)
        timesteps_this_batch = 0
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            # animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                # if animate_this_episode:
                #     env.render()
                #     time.sleep(0.05)
                obs.append(ob)
                ac = self.session.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None]})
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > self.max_path_length:
                    break
            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs)}
            self.paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        env.close()
        self.total_timesteps = timesteps_this_batch


#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             num_threads_gen=1,
             multi_steps_gd=1,
             reuse_nn_bl=False
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    # 
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    tf.reset_default_graph()
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)


    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#

    if discrete:
        # YOUR_CODE_HERE
        sy_logits_na = build_mlp(sy_ob_no, ac_dim, "nn", n_layers=n_layers, size=size)

        # Hint: Use the tf.multinomial op
        # the shape -1 automatically infers that the reshape will be done in the None axis
        sy_sampled_ac = tf.reshape(tf.multinomial(sy_logits_na, 1), shape=[-1])
        # negative in front is to remove the negative nature of cross entropy
        sy_logprob_n = - tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sy_logits_na,
                                                                        labels=sy_ac_na)

    else:
        # YOUR_CODE_HERE
        sy_mean = build_mlp(sy_ob_no, ac_dim, "nn", n_layers=n_layers, size=size)
        # logstd should just be a trainable variable, not a network output.
        sy_logstd = tf.get_variable('logstd', shape=[1, ac_dim], dtype=tf.float32, initializer=tf.zeros_initializer)

        sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal(tf.shape(sy_mean))

        # Hint: Use the log probability under a multivariate gaussian.
        sy_z = (sy_ac_na - sy_mean) / tf.exp(sy_logstd)
        sy_logprob_n = -0.5 * tf.reduce_sum(tf.square(sy_z), axis=1)

        # sy_logprob_n = - 1/2 * tf.nn.l2_loss(sy_mean - sy_ac_na)



    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    # Loss function that we'll differentiate to get the policy gradient.
    # Negative is to maximize the loss, instead of minimizing
    loss = - tf.reduce_mean(sy_logprob_n * sy_adv_n)
    update_op = tf.train.AdamOptimizer(learning_rate, name='AdamPolicy').minimize(loss)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        if not reuse_nn_bl:
            baseline_prediction = tf.squeeze(build_mlp(
                                    sy_ob_no,
                                    1,
                                    "nn_baseline",
                                    n_layers=n_layers,
                                    size=size))
        else:
            baseline_prediction = tf.squeeze(build_mlp(
                                    sy_ob_no,
                                    1,
                                    "nn_baseline",
                                    n_layers=n_layers,
                                    size=size,
                                    reuse_hidden_layers=True,
                                    reuse_scope_name="nn"))
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        # YOUR_CODE_HERE
        sy_target_bn = tf.placeholder(tf.float32, shape=[None], name='target_bn')
        loss_bn = tf.nn.l2_loss(sy_target_bn - baseline_prediction)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate, name='AdamBL').minimize(loss_bn)


    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        paths = []
        gen_start_time = time.time()
        if num_threads_gen == 1:
            # Collect paths until we have enough timesteps
            timesteps_this_batch = 0
            while True:
                ob = env.reset()
                obs, acs, rewards = [], [], []
                animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
                steps = 0
                while True:
                    if animate_this_episode:
                        env.render()
                        time.sleep(0.05)
                    obs.append(ob)
                    ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
                    ac = ac[0]
                    acs.append(ac)
                    ob, rew, done, _ = env.step(ac)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > max_path_length:
                        break
                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs)}
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                if timesteps_this_batch > min_timesteps_per_batch:
                    break
            total_timesteps += timesteps_this_batch
        else:
            # Multithread approach using tf coordinator

            coord = tf.train.Coordinator()

            workers = [TrajectionRunner(sess,
                                        sy_sampled_ac, sy_ob_no,
                                        env_name,
                                        max_path_length, min_timesteps_per_batch // num_threads_gen)
                       for _ in range(num_threads_gen)]

            for wrk in workers:
                wrk.start()

            coord.join(workers)

            # After here, all workers should be ready, let's collect their data

            timesteps_this_batch = 0
            for wrk in workers:
                paths.extend(wrk.paths)
                timesteps_this_batch = wrk.total_timesteps
                total_timesteps += wrk.total_timesteps

        gen_total_time = time.time() - gen_start_time
        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        # ====================================================================================#

        # YOUR_CODE_HERE
        # wrong, every path leads to different rewards!

        def discount_rewards(rwds, rtg):
            q = np.zeros_like(rwds)
            s = 0
            for t in reversed(range(rwds.shape[0])):
                s = s * gamma + rwds[t]
                q[t] = s

            if not rtg:
                q[:] = q[0]
            return q

        q_n = np.concatenate([discount_rewards(path["reward"], reward_to_go) for path in paths])

        # ====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        # ====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no})
            b_n = rescale(normalize(b_n), q_n.mean(axis=0, keepdims=True), q_n.std(axis=0, keepdims=True))

            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            # YOUR_CODE_HERE
            adv_n = normalize(adv_n)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            norm_q_n = normalize(q_n)
            total_bn_loss = 0
            for _ in range(multi_steps_gd):
                _, bn_loss = sess.run([baseline_update_op, loss_bn], feed_dict={sy_ob_no: ob_no, sy_target_bn: norm_q_n})
                total_bn_loss += bn_loss
            total_bn_loss /= multi_steps_gd
        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        total_loss = 0
        for _ in range(multi_steps_gd):
            _, current_loss = sess.run([update_op, loss], feed_dict={sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
            total_loss += current_loss
        total_loss /= multi_steps_gd

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("GenTime", gen_total_time)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("Loss", total_loss)
        if nn_baseline:
            logz.log_tabular("BNLoss", total_bn_loss)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--num_threads', '-nt', type=int, default=1)
    parser.add_argument('--multi_steps', '-ms', type=int, default=1)
    parser.add_argument('--reuse_nn_bl', '-rnb', action='store_true')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                num_threads_gen=args.num_threads,
                multi_steps_gd=args.multi_steps,
                reuse_nn_bl=args.reuse_nn_bl
                )

        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        # p = Process(target=train_func, args=tuple())
        # p.start()
        # p.join()
        train_func()
        

if __name__ == "__main__":
    main()
