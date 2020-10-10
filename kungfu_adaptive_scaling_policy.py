import time

import kungfu.tensorflow as kf
import numpy as np
import tensorflow as tf
from kungfu.python import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import resize
from kungfu.tensorflow.policy import BasePolicy
from kungfu.tensorflow.variables import GraphKeys


class AdaptiveScalingPolicy(BasePolicy):
    def __init__(self, batch_size, max_workers, num_steps, change_step, alpha):
        self._batch_size = batch_size
        self._average_throughput = dict()
        self._stop_scaling = False
        self._change_step = change_step
        self._throughputs = np.zeros(self._change_step)
        self._output = np.zeros((num_steps + 1, 6))
        self._alpha = alpha
        self._max_workers = max_workers

        self._need_sync = True

    def before_train(self):
        self._size_place = tf.placeholder(dtype=tf.uint32, shape=[])
        self._resize_op = resize(self._size_place)
        self._sync_op = BroadcastGlobalVariablesOp()

    def before_epoch(self, sess):
        pass

    def before_step(self, sess):
        self._start_time = time.time()

        if self._need_sync:
            print('running sync')
            sess.run(self._sync_op)
            print('finish sync')
            self._need_sync = False

    def after_step(self, sess):
        global_step = sess.run(tf.train.get_global_step())
        now = time.time()
        duration = now - self._start_time
        sub_step = global_step % self._change_step
        self._throughputs[sub_step] = self._batch_size / duration
        num_workers = current_cluster_size()

        if sub_step == 0 and not self._stop_scaling:
            self._average_throughput[num_workers] = np.mean(
            self._throughputs[self._change_step // 2:]) * num_workers
            print("global_step", global_step, "average_throughput",
                  self._average_throughput[num_workers], "number of workers",
                  num_workers)
            if num_workers > 1 and (num_workers -
                                    1) in self._average_throughput:
                last_throughput = self._average_throughput[num_workers - 1]
            else:
                last_throughput = 0
            if self._average_throughput[num_workers] < (
                    1 + (self._alpha * 1 / num_workers)) * last_throughput:
                self._stop_scaling = True
                print("stop scaling")
                new_cluster_size = num_workers - 1
                print('resize down to %s' % (new_cluster_size))
                self._need_sync = self._resize(sess, new_cluster_size)
                print('after resize down to %s, need sync: %s' % (new_cluster_size, self._need_sync))
            else:
                if num_workers <= self._max_workers:
                    new_cluster_size = num_workers + 1
                    print('resize up to %s' % (new_cluster_size))
                    self._need_sync = self._resize(sess, new_cluster_size)
                    print('after resize up to %s, need sync: %s' % (new_cluster_size, self._need_sync))
                else:
                    self._stop_scaling = True
                    print("stop scaling")

        after_run_duration = time.time() - now
        self._output[global_step] = [
            global_step, sub_step, num_workers, duration,
            self._throughputs[sub_step], after_run_duration
        ]

    def _resize(self, sess, new_size):
        return sess.run(self._resize_op,
                        feed_dict={
                            self._size_place: new_size,
                        })

    def after_train(self, sess):
        fname = "out_{}.csv".format(current_rank())
        header = "global_step,sub_step,num_workers,duration,throughput,after_run_duration"
        np.savetxt(fname, self._output, delimiter=",", header=header)
