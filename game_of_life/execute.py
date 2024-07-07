from game_of_life.game_of_life import simulate_process
import numpy as np


simulate_process('alive.map', 'result', 'backend', 'game_of_life.onnx', 100)
