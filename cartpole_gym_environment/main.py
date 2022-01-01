# https://tonari-it.com/python-pycharm-package-install/
import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt
import gym
import socket
import re
from multiprocessing import Pool
import typing
from JSAnimation.IPython_display import display_animation
from matplotlib import animation


def tcp_server(args):
    i, socket, env = args
    env.reset()
    socket.bind(("127.0.0.1", 8080 + i))
    socket.listen(10)
    client, _ = socket.accept()

    print("connected %d" % i)
    while True:
        if i == 0: env.render(mode='rgb_array')
        msg = client.recv(1024).decode('ascii')
        if msg == 'reset':
            observation = env.reset()
            client.send(
                bytes("o:%f,%f,%f,%f\n" % (observation[0], observation[1], observation[2], observation[3])
                      , encoding="ascii"))
        else:
            action = int(msg)
            observation, reward, isDone, info = env.step(action)
            client.send(
                bytes("r:%f,%f,%f,%f,%f,%f\n"% (observation[0], observation[1], observation[2], observation[3], reward, isDone)
                , encoding="ascii"))
    return 0


def gym_environment(n: int):
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for i in range(n)]
    envs = [gym.make('CartPole-v0') for i in range(n)]
    p = Pool(32)
    values = [(i, sockets[i], envs[i]) for i in range(n)]
    p.map(tcp_server, values)
    p.close()


if __name__ == '__main__':
    # gym_environment(1) # duel
    gym_environment(32)
