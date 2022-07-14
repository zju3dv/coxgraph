#!/usr/bin/env python

from logging import INFO
import sys
import rospy
import rosnode
import rosgraph
import re
import time
import datetime
import glog
import os
import evaluator as evaluator
import eval_plotting as plotting


ID = '/rosnode'


class Evaluator:
    def __init__(self):
        self.node_eval_threads = {}
        self.topic_eval_threads = {}
        self.eval_rate_s = rospy.get_param('~eval_rate_s', default=0.5)

        self.node_names = rospy.get_param('~node_names')
        for i in range(0, len(self.node_names)):
            self.node_names[i] = rosgraph.names.script_resolve_name(
                '/', self.node_names[i])
        node_eval_mode = rospy.get_param('~node_eval_mode')
        if self.node_names is not None or node_eval_mode is not None:
            glog.check_eq(len(self.node_names), len(node_eval_mode))
            self.eval_mode = {}
            for name, mode in zip(self.node_names, node_eval_mode):
                self.eval_mode[name] = mode
            for node_name in self.node_names:
                self.node_eval_threads[node_name] = {}
        self.plot_dir = os.path.join(rospy.get_param('~plot_dir', '.'), datetime.datetime.now().strftime(
            '%x').replace('/', '-')+'-'+datetime.datetime.now().strftime('%X').replace(':', '-'))
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        print("Saving results to "+self.plot_dir)

        self.topic_names = rospy.get_param('~topic_names', default=None)
        topic_eval_mode = rospy.get_param('~topic_eval_mode', default=None)
        if self.topic_names is not None or self.topic_names is not None:
            glog.check_eq(len(self.topic_names), len(topic_eval_mode))
            self.topic_eval_mode = {}
            for name, mode in zip(self.topic_names, topic_eval_mode):
                self.topic_eval_mode[name] = mode
            for topic in self.topic_names:
                self.topic_eval_threads[topic] = {}

        self.sys_eval_mode = rospy.get_param('~sys_eval_mode', default=None)
        self.sys_eval_threads = {}

        self.master = rosgraph.Master(ID)
        self.node_pid = {}
        self.plot_threads = {}
        self.start_eval()

    def start_eval(self):
        rate = rospy.Rate(2)
        while self.node_names is not None and not rospy.is_shutdown():
            rate.sleep()
            all_node_names = rosnode.get_node_names()
            for node_name in self.node_names:

                # check if node is running
                if node_name not in all_node_names:
                    rospy.logwarn('Node %s is not running' % node_name)
                    continue

                if node_name in self.node_pid:
                    continue

                rospy.loginfo('Looking for pid of node %s' % node_name)
                node_api = rosnode.get_api_uri(self.master, node_name)
                while True:
                    try:
                        node_con_info = rosnode.get_node_connection_info_description(
                            node_api, self.master)
                    except rosnode.ROSNodeIOException as e:
                        time.sleep(0.1)
                        rospy.loginfo_throttle(1, e)
                        continue
                    else:
                        break
                pid_match = re.search('Pid: (\d+)', node_con_info)
                if pid_match is None:
                    rospy.logwarn('Not found pid in description of node %s' %
                                  node_name)
                    continue
                self.node_pid[node_name] = int(pid_match.group(1))
                rospy.loginfo('Pid: %d' % self.node_pid[node_name])

            if len(self.node_pid) == len(self.node_names):
                break

        rospy.loginfo('Catched pid of every node, start evaluating')

        self._start_eval_threads()

        rospy.on_shutdown(self.stop_threads)

        self._plot_loop()

    def _start_eval_threads(self):
        if self.node_names is not None:
            for node_name in self.node_names:
                for eval_mode in self.eval_mode[node_name]:
                    eval_thread = evaluator.EvaluatorFactory.create_evaluator(
                        eval_mode,
                        node_name=node_name,
                        node_pid=self.node_pid[node_name],
                        eval_rate_s=self.eval_rate_s)
                    eval_thread.start()
                    self.node_eval_threads[node_name][eval_mode] = eval_thread
                    self._add_to_plotting(eval_mode, eval_thread)

        if self.topic_names is not None:
            for topic in self.topic_names:
                for eval_mode in self.topic_eval_mode[topic]:
                    eval_thread = evaluator.EvaluatorFactory.create_evaluator(
                        eval_mode,
                        topic=topic,
                        eval_rate_s=self.eval_rate_s)
                    eval_thread.start()
                    self.topic_eval_threads[topic][eval_mode] = eval_thread
                    self._add_to_plotting(eval_mode, eval_thread)

        if self.sys_eval_mode is not None:
            for eval_mode in self.sys_eval_mode:
                eval_thread = evaluator.EvaluatorFactory.create_evaluator(
                    eval_mode,
                    eval_rate_s=self.eval_rate_s)
                eval_thread.start()
                self.sys_eval_threads[eval_mode] = eval_thread
                self._add_to_plotting(eval_mode, eval_thread)

        for plot_mode in self.plot_threads:
            self.plot_threads[plot_mode].start()

    def _add_to_plotting(self, eval_mode, eval_thread):
        if eval_mode == 'sys_bw' or eval_mode == 'bw_from_msg':
            eval_mode = 'topic_bw'
        if eval_mode not in self.plot_threads:
            self.plot_threads[eval_mode] = plotting.PlottingFactory.create_plotting(
                eval_mode, plot_dir=self.plot_dir, plot_rate_s=1.0)
        self.plot_threads[eval_mode].add_stat_update_callback(
            eval_thread.get_eval_stat)

    def _plot_loop(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            for plot_mode in self.plot_threads:
                self.plot_threads[plot_mode].plot()
                time_to_sleep = start_time + self.eval_rate_s - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                start_time = time.time()

    def stop_threads(self):
        for node_name in self.node_names:
            for eval_mode in self.eval_mode[node_name]:
                self.node_eval_threads[node_name][eval_mode].stop()

        for topic in self.topic_eval_threads:
            for eval_mode in self.topic_eval_threads[topic]:
                self.topic_eval_threads[topic][eval_mode].stop()

        for key in self.plot_threads:
            self.plot_threads[key].stop()


if __name__ == "__main__":
    rospy.init_node('evaluator', anonymous=True)
    evaluator = Evaluator()
    rospy.spin()
