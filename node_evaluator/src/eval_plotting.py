from matplotlib import pyplot as plt
import os
import csv
import threading
from glog import logging
import numpy as np


COLOR_MAP = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-']


class PlottingFactory(object):

    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print('Plotting %s already exists.' % name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_plotting(cls, name, **kwargs):
        if name not in cls.registry:
            print('Plotting %s does not exists.')
            raise NotImplementedError

        plot_class = cls.registry[name]
        plotting = plot_class(**kwargs)
        return plotting

    def __init__(self):
        pass


class PlottingBase(threading.Thread):
    def __init__(self, **kwargs):
        super(PlottingBase, self).__init__()
        self.plot_dir = kwargs['plot_dir']
        self.plot_rate_s = kwargs['plot_rate_s']
        self.stat_update_callback = []
        self.stat_update_lock = threading.Lock()
        self.term_event = threading.Event()
        self.eval_stat = {}
        self.color_map = {}

    def add_stat_update_callback(self, stat_cb):
        self.stat_update_callback.append(stat_cb)

    def run(self):
        logging.info('Start %s plotting' % self.plot_mode)
        threading.Timer(self.plot_rate_s, self._stat_update_loop).start()
        self.term_event.wait()
        logging.info('%s plotting stopped' % self.plot_mode)

    def _plot_loop(self):
        if not self.term_event.is_set():
            self.plot()
            threading.Timer(self.plot_rate_s, self._plot_loop).start()

    def plot(self):
        plt.figure(num=self.plot_mode)
        self._plot()

    def _plot(self):
        pass

    def _stat_update_loop(self):
        if not self.term_event.is_set():
            with self.stat_update_lock:
                self.stat_update()
            threading.Timer(self.plot_rate_s, self._stat_update_loop).start()

    def stat_update(self):
        for update in self.stat_update_callback:
            new_stat = update()
            for key in new_stat:
                if 'time' in new_stat:
                    if key is not 'time' and len(new_stat[key]) > 0:
                        self.eval_stat[key] = new_stat
                else:
                    if len(new_stat[key][key]) > 0:
                        self.eval_stat[key] = new_stat[key]
                if key not in self.color_map and key is not 'time':
                    self.color_map[key] = COLOR_MAP[len(self.eval_stat)-1]

    def stop(self):
        self.term_event.set()
        with open(os.path.join(self.plot_dir, self.plot_mode+".csv"), "w") as csv_file:
            writer = csv.writer(csv_file)
            for k, v in self.eval_stat.items():
                writer.writerow([k, v])


@PlottingFactory.register('cpu')
class CPUPlotting(PlottingBase):
    def __init__(self, **kwargs):
        super(CPUPlotting, self).__init__(**kwargs)
        self.plot_mode = 'cpu'

    def _plot(self):
        x = {}
        y = {}
        curve_names = []
        for key in self.eval_stat:
            with self.stat_update_lock:
                x[key] = np.array(self.eval_stat[key]['time'])
                y[key] = np.array(self.eval_stat[key][key])
            plt.plot(x[key], y[key], self.color_map[key])
            curve_names.append(key)
        plt.title('CPU Usage')
        plt.xlabel('time(s)')
        plt.ylabel('cpu usage(%)')
        plt.legend(curve_names, loc='upper left', fancybox=True)
        plt.savefig(os.path.join(self.plot_dir, '%s.png' % self.plot_mode))


@PlottingFactory.register('mem')
class MemPlotting(PlottingBase):
    def __init__(self, **kwargs):
        super(MemPlotting, self).__init__(**kwargs)
        self.plot_mode = 'mem'

    def _plot(self):
        x = {}
        y = {}
        curve_names = []
        for key in self.eval_stat:
            with self.stat_update_lock:
                x[key] = np.array(self.eval_stat[key]['time'])
                y[key] = np.array(self.eval_stat[key][key])
            plt.plot(x[key], y[key], self.color_map[key])
            curve_names.append(key)
        plt.title('Memory Usage')
        plt.xlabel('time(s)')
        plt.ylabel('memory usage(%)')
        plt.legend(curve_names, loc='upper left', fancybox=True)
        plt.savefig(os.path.join(self.plot_dir, '%s.png' % self.plot_mode))


@PlottingFactory.register('topic_bw')
class TopicBwPlotting(PlottingBase):
    def __init__(self, **kwargs):
        super(TopicBwPlotting, self).__init__(**kwargs)
        self.plot_mode = 'topic_bw'

    def _plot(self):
        x = {}
        y = {}
        curve_names = []
        if len(self.eval_stat) == 0:
            return
        for key in self.eval_stat:
            with self.stat_update_lock:
                x[key] = np.array(self.eval_stat[key]['time'])
                y[key] = np.array(self.eval_stat[key][key])
            plt.plot(x[key], y[key], self.color_map[key])
            curve_names.append(key)
        plt.title('Bandwidth')
        plt.xlabel('time(s)')
        plt.ylabel('bandwith(MB/s)')
        plt.legend(curve_names, loc='upper left', fancybox=True)
        plt.savefig(os.path.join(self.plot_dir, '%s.png' % self.plot_mode))
