import tensorflow as tf
import time
import json
import numpy as np
import urllib
import os
from kungfu.python import current_rank, current_cluster_size

class ScalingHook(tf.train.SessionRunHook):
    def __init__(self, batch_size, num_training_steps):
        self._batch_size = batch_size
        self._average_throughput = dict()
        path = "scaling_workers.json"
        self._workers = self.read_workers(path)
        self._stop_scaling = False
        self._change_step = 100
        self._throughputs = np.zeros(self._change_step)
        self._num_training_steps = num_training_steps
        self._output = np.zeros((num_training_steps + 1, 6))
        self._alpha = 0.33

    def begin(self):
        self._global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        if current_rank() == 0:
            self._start_time = time.time()

    def after_run(self, run_context, run_values):
        if current_rank() == 0:
            now = time.time()
            duration = now - self._start_time
            global_step = run_context.session.run(self._global_step)
            sub_step = global_step % self._change_step
            self._throughputs[sub_step] = self._batch_size / duration
            num_workers = current_cluster_size()
            if sub_step == 0 and not self._stop_scaling:
                self._average_throughput[num_workers] = np.mean(self._throughputs[self._change_step//2:]) * num_workers
                print("global_step", global_step, "average_throughput", self._average_throughput[num_workers], "number of workers", num_workers)
                last_throughput = self._average_throughput[num_workers - 1] if num_workers > 1 else 0
                if self._average_throughput[num_workers] < (1 + (self._alpha * 1/num_workers)) * last_throughput:
                    self._stop_scaling = True
                    print("stop scaling")
                    self.remove_last_worker(num_workers)
                else:
                    if num_workers < len(self._workers) + 1:
                        self.add_worker(num_workers)
                    else:
                        self._stop_scaling = True
                        print("stop scaling")
            after_run_duration = time.time() - now
            self._output[global_step] = [global_step, sub_step, num_workers, duration, self._throughputs[sub_step], after_run_duration]
            if global_step == self._num_training_steps:
                fname = "out.csv"
                np.savetxt(fname, self._output, delimiter=",", header="global_step,sub_step,num_workers,duration,throughput,after_run_duration")

    def end(self, session):
        if current_rank() == 0:
            fname = "out.csv"
            np.savetxt(fname, self._output, delimiter=",", header="global_step,sub_step,num_workers,duration,throughput,after_run_duration")

    def read_workers(self, path):
        with open(path, "r") as json_file:
            data = json_file.read()
        return json.loads(data)

    def add_worker(self, num_workers, url="http://127.0.0.1:9100/addworker"):
        worker = self._workers[num_workers - 1]
        data = json.dumps(worker).encode("utf-8")
        req =  urllib.request.Request(url, data=data, method="POST")
        resp = urllib.request.urlopen(req)
        if resp.getcode() != 200:
            print("request failed")

    def remove_last_worker(self, num_workers, url="http://127.0.0.1:9100/removeworker"):
        try:
            worker = self._workers[num_workers - 2]
        except:
            print("cannot remove the only worker")
            return
        data = json.dumps(worker).encode("utf-8")
        req =  urllib.request.Request(url, data=data, method="POST")
        resp = urllib.request.urlopen(req)
        if resp.getcode() != 200:
            print("request failed")
