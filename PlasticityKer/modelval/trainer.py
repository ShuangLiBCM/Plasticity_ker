"""
    Create network classes
"""
import tensorflow as tf
from os.path import join
from os import path, makedirs
from tensorflow.python.util import nest

class Trainer(object):

    def __init__(self, mse, total_loss, input_name, target_name=None, session=None, save_dir=None, optimizer_op=tf.train.AdamOptimizer,
                 optimizer_config={}, device_config=['/device:GPU:0']):   # If ground truth kernel is given ,it will be used.
        self.loss = mse
        self.total_loss = total_loss
        self.inputs_ = input_name
        self.targets_ = target_name
        self.graph = total_loss.graph
        self.session = session
        self.optimizer_op = optimizer_op
        self.device_config = device_config
        self.optimizer_config = optimizer_config
        self.loss_tracker = []      # Used for tracking the loss
        self.save_dir = save_dir
        self.build()
        self.init_session()

    def build(self):

        with self.graph.as_default():

            for d in self.device_config:
                with tf.device(d):

                    self.optimizer = self.optimizer_op(**self.optimizer_config)
                    self.global_step = tf.Variable(0, trainable=False,
                                                   name='global_step')  # Count how many times the network got updated

                    self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

                    # Configure to load all variables except for the global step variable

                    variable_to_load = [v for v in tf.global_variables() if 'global_step' not in v.name]
                    self.saver = tf.train.Saver(max_to_keep=10)        # Save the whole session
                    self.saver_best = tf.train.Saver(variable_to_load, max_to_keep=1)      # Save only the variable
                    self.init = tf.global_variables_initializer()

    def init_session(self):

        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
        self.mini_vali_loss = None

        with self.graph.as_default():
            self.session.run(self.init)

    def make_feed_dict(self, inputs=None, targets=None, feed_dict=None):

        if feed_dict is not None:
            fd = dict(feed_dict)
        else:
            fd = {}

        if inputs is not None:
            nest.assert_same_structure(self.inputs_, inputs)
            inputs_ph_list = nest.flatten(self.inputs_)
            inputs_list = nest.flatten(inputs)

            for ph, val in zip(inputs_ph_list, inputs_list):
                fd[ph] = val

        # feed in targets
        if self.targets_ is not None and targets is not None:
            nest.assert_same_structure(self.targets_, targets)
            targets_ph_list = nest.flatten(self.targets_)
            targets_list = nest.flatten(targets)

            for ph, val in zip(targets_ph_list, targets_list):
                fd[ph] = val

        fd = {k: v for k, v in fd.items() if v is not None}

        return fd

    def train(self, train_data, vali_data, batch_size=256, save_model_freq=-1, vali_freq=50, min_error=-1, burn_in_steps=100,
              early_stopping_steps=15, max_steps=-1, load_best=True, feed_dict=None):
        """
        Train the network until early stopping or maximum training number achieved.
        :param train_data: Input data used for training
        :param vali_data:  Input data used for validation
        :param batch_size: batch size for training
        :param save_model_freq: Number of training steps between saving the model
        :param vali_freq:  Number of steps between model validation
        :param min_error: minimum error for stopping training
        :param burn_in_steps: Number of initial training steps without model validation
        :param early_stopping_steps: Number of non-improved validation steps before terminating the training
        :param max_steps: Maximum of number of training , if max_steps <=0, no maximum training number is enforced
        :param load_best: Whether to load the best model at the end of the training
        :param feed_dict: External feed_dict for extra hyperparameter
        :return:
        """
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
            
        with self.graph.as_default():

            burn_in_steps = burn_in_steps if burn_in_steps is not None else vali_freq * 4      # Prepare training steps before validation
            # Save the model before training
            checks_without_update = 0  # Numbers of non-improved validation score check
            sess = self.session

            if self.mini_vali_loss is None:

                x_vali_batch, y_vali_batch = vali_data.batch(batch_size)
                fd = self.make_feed_dict(x_vali_batch, y_vali_batch, feed_dict=feed_dict)
                vali_loss = sess.run(self.loss, feed_dict=fd)
                print('\nInitial validation mse=%.5f' % vali_loss, flush=True)
                self.mini_vali_loss = vali_loss
                self.save_best()
                self.save(0)    # Save the initial model

            step = 1

            while step < max_steps or max_steps <= 0:
                if (step % save_model_freq == 0) & (save_model_freq > 0):
                    self.save(step)

                x_train_batch, y_train_batch = train_data.next_batch(batch_size)
                fd = self.make_feed_dict(x_train_batch, y_train_batch, feed_dict=feed_dict)
                _, global_step = sess.run([self.train_step, self.global_step], feed_dict=fd)

                if global_step > burn_in_steps and global_step % vali_freq == 0:

                    x_train_batch, y_train_batch = train_data.batch(batch_size)
                    fd = self.make_feed_dict(x_train_batch, y_train_batch, feed_dict=feed_dict)
                    train_loss = sess.run(self.loss, feed_dict=fd)

                    x_vali_batch, y_vali_batch = vali_data.x, vali_data.y    # Use the whole validation set
                    fd = self.make_feed_dict(x_vali_batch, y_vali_batch, feed_dict=feed_dict)
                    vali_loss = sess.run(self.loss, feed_dict=fd)
                    self.loss_tracker.append([train_loss, vali_loss])
                    print('Global Step %04d and Step %04d: validation mse=%.5f' % (global_step, step, vali_loss), flush=True)

                    # Perform early stopping
                    if (min_error < 0) | (vali_loss > min_error):
                        if vali_loss < self.mini_vali_loss:
                            self.mini_vali_loss = vali_loss
                            print('Updated min validation loss!Saving model...')
                            self.save_best()
                            checks_without_update = 0
                        else:
                            checks_without_update += 1
                            if checks_without_update == early_stopping_steps:
                                print('Early Stopping!!!')
                                break
                    else:
                        print('Minimum error reached!!!')
                        break

                step += 1

            if load_best:
                print('Restoring the best parameters...')
                self.restore_best()

        return self.mini_vali_loss

    def evaluate(self, inputs=None, targets=None, ops=None, feed_dict=None):
        """
        Evaluate the network on OPS
        :param inputs: inputs used to create the feed_dict
        :param targets: targets used to create the feed_dict
        :param ops: operation to evaluated
        :param feed_dict: other parameters
        :return:
        """
        if ops is None:
            ops = self.loss
        fd = self.make_feed_dict(inputs=inputs, targets=targets, feed_dict=feed_dict)
        return self.session.run(ops, feed_dict=fd)

    def save(self, step=None):
        """
        Save current state of the graph into a checkpoint file.
        :param step: step number to tag the checkpoint file with. If not given, defaults to the value of global_step variable.
        """
        if step is None:
            step = self.session.run(self.global_step)
        self.saver.save(self.session, join(self.save_dir, 'step'), global_step=step)

    def save_best(self):
        """
        Save the current state of the graph as best
        :return:
        """
        self.saver_best.save(self.session, join(self.save_dir, 'best'))

    def restore_best(self):

        """
        Restore the last saved "best" state of the gragh.
        :return:
        """
        self.saver_best.restore(self.session, join(self.save_dir, 'best'))
