import os
from sysv_ipc import SharedMemory
from threading import Thread

import cv2
from gym.envs.registration import registry, register, make, spec, EnvSpec
from subprocess import Popen

import subprocess

import gym
import math

import time
import numpy as np
from gym import spaces
from gym.envs import registration
from gym.utils import seeding
from IPython import embed

import snakeoil
import xserver_util
from logger import Logger
from xserver_util import get_screen


class ProcessLogPoller(Thread):
    def __init__(self, logfile):
        Thread.__init__(self)
        self.logfile = os.path.join(logfile, 'torcs.log')
        self.process = None

    def set_process(self, process):
        self.process = process

    def run(self):
        while True:
            if self.process:
                line = self.process.stdout.readline()
                if line:
                    with open(self.logfile, 'a') as fw:
                        fw.write('Pid:{}, output:{}\n'.format(self.process.pid, line))
            else:
                time.sleep(1)


class TorcsEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    min_steps = 1000
    min_speed = 5  # speed should not be less than 5km/h after min_steps

    ACTIONS = [('steer', -1), ('steer', 0), ('steer', 1), ('accel', 0), ('accel', 0.5), ('accel', 1), ('brake', 0),
               ('brake', 0.5), ('brake', 1)]
    spec = EnvSpec('Torcs-v0', max_episode_steps=10000, reward_threshold=20000.0)

    class KEY_MAP:
        ENTER = 'KP_Enter'
        UP = 'KP_Up'

    LAUNCH_SEQ = [KEY_MAP.ENTER, KEY_MAP.ENTER, KEY_MAP.UP, KEY_MAP.UP, KEY_MAP.ENTER, KEY_MAP.ENTER]

    def __init__(self, env_id, width=640, height=480, frame_skip=(2, 5), torcs_dir='/usr/local',
                 logdir='/data/logs', **kwargs):
        self.env_id = env_id
        self.port = 9200 + self.env_id
        self.disp_name = ':{}'.format(20 + self.env_id)
        self.screen_w, self.screen_h = width, height
        self.torcs_dir = torcs_dir
        self.viewer = None
        self.time_step = 0
        self.client = None
        self.torcs_process = None
        self.shared_memory = None
        self.frameskip = frame_skip
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_h, self.screen_w, 3))
        self.reward_range = (-1, 50)
        self._seed()
        self.logdir = os.path.join(logdir, 'torcs-{}'.format(env_id))
        self.logger = Logger(self.env_id, self.logdir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def __start_torcs(self):
        if self.torcs_process:
            subprocess.call('rm -rf /tmp/.X*', shell=True)
            self.torcs_process.kill()
            self.disp_process.kill()
        else:
            self.torcs_logpoller = ProcessLogPoller(self.logdir)
            self.torcs_logpoller.start()

        color_depth = 24
        screen_res = '{}x{}x{}'.format(self.screen_w, self.screen_h, color_depth)
        self.disp_process = subprocess.Popen(['Xvfb', self.disp_name, '-ac', '-screen', '0', screen_res],
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        self.logger.info('Xvfb pid:{}, display:{}', self.disp_process.pid, self.disp_name)
        torcs_bin = os.path.join(self.torcs_dir, 'lib/torcs/torcs-bin')
        torcs_lib = os.path.join(self.torcs_dir, 'lib/torcs/lib')
        torcs_data = os.path.join(self.torcs_dir, 'share/games/torcs/')
        self.torcs_process = Popen([torcs_bin, '-port', str(self.port), '-nofuel', '-nodamage', '-nolaptime'],
                                   env={'LD_LIBRARY_PATH': torcs_lib, 'DISPLAY': self.disp_name},
                                   cwd=torcs_data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.torcs_logpoller.set_process(self.torcs_process)
        for cmd in self.LAUNCH_SEQ:
            time.sleep(0.8)
            subprocess.call(['xdotool', 'key', cmd], env={'DISPLAY': self.disp_name})
        self.shared_memory = SharedMemory(self.port)
        self.logger.info('Started torcs pid:{}, port:{}', self.torcs_process.pid, self.port)

    def __to_torcs_action(self, action):
        torcs_actions = {a[0]: 0.0 for a in self.ACTIONS}
        torcs_actions[self.ACTIONS[action][0]] = self.ACTIONS[action][1]
        return torcs_actions

    def _step(self, action):
        torcs_action = self.__to_torcs_action(action)

        self.client.R.d.update(torcs_action)
        self.client.R.d['gear'] = 1
        if self.client.S.d['speedX'] > 50:
            self.client.R.d['gear'] = 2

        if self.client.S.d['speedX'] > 80:
            self.client.R.d['gear'] = 3

        reward = 0.0
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])

        self.logger.info('Sending command to server:{}', self.client.S.d)
        for i in range(num_steps):
            self.client.respond_to_server()
            self.client.get_servers_input()
            speed = self.client.S.d['speedX']
            angle = self.client.S.d['angle']
            reward += speed * math.cos(angle)

        long_speed = speed * math.cos(angle)  # speed along x-axis

        track = np.array(self.client.S.d['track'])

        if track.min() < 0 or np.cos(angle) < 0 or self.time_step > self.min_steps and long_speed < 5:
            reward = -500
            done = True
            self.logger.debug('Terminal state!! track:{}, angle:{}, speed:{}, steps:{}', track, angle, long_speed,
                              self.time_step)
        else:
            reward = long_speed
            done = False

        self.image = xserver_util.get_screen_shm(self.shared_memory, self.screen_w, self.screen_h)
        info = {}
        if self.time_step % 5 == 0 or done:
            filename = os.path.join(self.logdir, '{}_{}.png'.format(self.torcs_process.pid, self.time_step))
            cv2.imwrite(filename, self.image)
        self.time_step += 1
        return self.image, reward, done, info

    def _reset(self):
        self.logger.info('Resetting torcs!!')
        while True:
            try:
                self.__start_torcs()
                self.client = snakeoil.Client(p=self.port)
                self.client.get_servers_input()
                break
            except Exception as e:
                self.logger.info("Failed to connect to torcs. Will try again.")
                continue

        self.image = get_screen(0, 0, self.screen_w, self.screen_h, self.disp_name)
        self.logger.info('Successfully reset torcs after timesteps:{}', self.time_step)
        self.time_step = 0
        return self.image

    def _render(self, mode='human', close=False):
        if close:
            self.torcs_process.kill()
            self.disp_process.kill()
        if mode == 'rgb_array':
            return self.image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.image)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    a = TorcsEnv(25, frame_skip=1)
    a.reset()
    a.reset()
    a.reset()
    print(a.action_space.n)
    for i in range(100000):
        ob, reward, done, info = a.step(5)
        if done:
            a.reset()
        # if i % 50 == 0:
        #     print('Resetting! Done.')
        #     a.reset()
        a.render()
        # cv2.imwrite('/home/sanjeev/debug/{}.png'.format(i), a.render('rgb_array'))
        # a.close()
