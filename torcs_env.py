import os
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
from xserver_util import get_screen
from logger import logger


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

    def __init__(self, env_id, width=640, height=480, frame_skip=(2, 5), torcs_dir='/usr/local'):
        self.env_id = env_id
        self.port = 9200 + self.env_id
        print('Port', self.port)
        self.disp_name = ':{}'.format(20 + self.env_id)
        self.screen_w, self.screen_h = width, height
        self.torcs_dir = torcs_dir
        self.__game_init()
        self.viewer = None
        self.time_step = 0
        self.client = None
        self.torcs_process = None
        self.frameskip = frame_skip
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_h, self.screen_w, 3))
        self.reward_range = (-1, 50)
        self._seed()

    def __game_init(self):
        color_depth = 24
        screen_res = '{}x{}x{}'.format(self.screen_w, self.screen_h, color_depth)
        self.disp_process = subprocess.Popen(['Xvfb', self.disp_name, '-ac', '-screen', '0', screen_res])

        logger.info('Xvfb pid:{}, display:{}', self.disp_process.pid, self.disp_name)

    def __start_torcs(self):
        torcs_bin = os.path.join(self.torcs_dir, 'lib/torcs/torcs-bin')
        torcs_lib = os.path.join(self.torcs_dir, 'lib/torcs/lib')
        torcs_data = os.path.join(self.torcs_dir, 'share/games/torcs/')
        self.torcs_process = Popen([torcs_bin, '-port', str(self.port), '-nofuel', '-nodamage', '-nolaptime'],
                                   env={'LD_LIBRARY_PATH': torcs_lib, 'DISPLAY': self.disp_name},
                                   cwd=torcs_data)
        for cmd in self.LAUNCH_SEQ:
            time.sleep(0.5)
            subprocess.call(['xdotool', 'key', cmd], env={'DISPLAY': self.disp_name})
        logger.info('Started torcs pid:{}, port:{}', self.torcs_process.pid, self.port)

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

        logger.info('Sending command to server:{}', self.client.S.d)
        for i in range(num_steps):
            self.client.respond_to_server()
            self.client.get_servers_input()
            speed = self.client.S.d['speedX']
            angle = self.client.S.d['angle']
            reward += speed * math.cos(angle)

        long_speed = speed * math.cos(angle)  # speed along x-axis

        track = np.array(self.client.S.d['track'])

        if track.min() < 0 or np.cos(angle) < 0 or self.time_step > self.min_steps and long_speed < 5:
            reward = -1
            done = True
            logger.debug('Terminal state!! track:{}, angle:{}, speed:{}, steps:{}', track, angle, long_speed,
                         self.time_step)
        else:
            reward = long_speed
            done = False

        self.image = get_screen(0, 0, self.screen_w, self.screen_h, self.disp_name)
        info = {}
        self.time_step += 1
        return self.image, reward, done, info

    def _reset(self):
        logger.info('Resetting torcs!!')
        if self.torcs_process:
            self.torcs_process.kill()
        self.__start_torcs()
        self.client = snakeoil.Client(p=self.port)
        self.client.get_servers_input()
        self.image = get_screen(0, 0, self.screen_w, self.screen_h, self.disp_name)
        logger.info('Successfully reset torcs after timesteps:{}', self.time_step)
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
    a = TorcsEnv(1)
    a.reset()
    print(a.action_space.n)
    for i in range(10000):
        ob, reward, done, info = a.step(a.action_space.sample())
        if i % 50 == 0:
            print('Resetting! Done.')
            a.reset()
            # a.render()
            # a.close()
