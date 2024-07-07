import numpy as np
import onnxruntime as ort
import onnx
import matplotlib.pyplot as plt
from pcraster import *
from pcraster.framework import *
import os


from .utils import execute_graph
from .custom_backend import operator_map 

class GameOfLife(DynamicModel):
    def __init__(self, input_map, output_map, model_path, model_type):
        DynamicModel.__init__(self)
        setclone(20, 20, 1, 0, 0)
        self.input_map = input_map
        self.output_map = output_map
        self.model_path = model_path
        self.model_type = model_type
        self.operator_map = operator_map  # Initialize the operator map correctly as a dictionary
        self.alive_counts = []
        self.alive_grids = []
        print(f"Initialized GameOfLife with model_type: {model_type}")

    def initial(self):
        print("Initial method started.")
        if not os.path.exists(self.input_map):
            raise FileNotFoundError(f"Input map file {self.input_map} not found.")
        
        try:
            self.alive = readmap(self.input_map)
        except RuntimeError as e:
            raise RuntimeError(f"Error reading input map file {self.input_map}: {e}")
        
        self.report(self.alive, self.output_map)
        self.alive = self.alive == 1
        self.defined = defined(self.alive)
        
        if self.model_type in ['runtime', 'backend']:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} not found.")
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.model = onnx.load(self.model_path)

        # Record initial alive count and grid
        self.alive_counts.append(np.sum(pcr2numpy(self.alive, 0)))
        self.alive_grids.append(pcr2numpy(self.alive, 0))

    def dynamic(self, output_type=Scalar):
        print(f'Running model type: {self.model_type}')
        if self.model_type == 'pcraster':
            self.run_pcraster_model()
        elif self.model_type == 'runtime':
            self.run_runtime_model(output_type)
        elif self.model_type == 'backend':
            self.run_backend_model(output_type)
        else:
            print(f"Unknown model_type: {self.model_type}")

        # Record alive count and grid at each step
        self.alive_counts.append(np.sum(pcr2numpy(self.alive, 0)))
        self.alive_grids.append(pcr2numpy(self.alive, 0))

    def run_pcraster_model(self):
        aliveScalar = scalar(self.alive)
        numberOfAliveNeighbours = windowtotal(aliveScalar, 3) - aliveScalar

        threeAliveNeighbours = numberOfAliveNeighbours == 3
        birth = pcrand(threeAliveNeighbours, pcrnot(self.alive))

        survivalA = pcrand((numberOfAliveNeighbours == 2), self.alive)
        survivalB = pcrand((numberOfAliveNeighbours == 3), self.alive)
        survival = pcror(survivalA, survivalB)
        self.alive = pcror(birth, survival)

        self.alive = ifthen(self.defined, self.alive)
        self.report(self.alive, self.output_map)

    def run_runtime_model(self, output_type):
        self.input_data = pcr2numpy(self.alive, 0).astype(np.float32).reshape(1, 20, 20, 1)
        ort_inputs = {self.input_name: self.input_data}
        ort_outs = self.ort_session.run(None, ort_inputs)
        self.alive_np = (ort_outs[0] > 0.5).astype(np.float32).reshape(20, 20)
        self.alive = numpy2pcr(output_type, self.alive_np, 241)
        self.alive = ifthen(self.defined, self.alive)
        self.report(self.alive, self.output_map)

    def run_backend_model(self, output_type):
        board = pcr2numpy(self.alive, 0).astype(np.float32).reshape(1, 20, 20, 1)
        output = execute_graph(self.model.graph, board)
        output_tensor_name = list(output.keys())[0]
        self.alive_np = (output[output_tensor_name] > 0.5).astype(np.float32).reshape(20, 20)
        self.alive = numpy2pcr(output_type, self.alive_np, 241)
        self.alive = ifthen(self.defined, scalar(self.alive))
        self.report(self.alive, self.output_map)

class GameOfLifeModel(DynamicFramework):
    def __init__(self, input_map, output_map, model_type, model_path, time_step):
        self.model = GameOfLife(input_map, output_map, model_path, model_type)
        DynamicFramework.__init__(self, self.model, time_step)
        self.model_type = model_type
        print(f"GameOfLifeModel initialized with model_type: {self.model_type}")

def run_game_of_life(input_map, output_map, model_type, model_path, time_step=5):
    game = GameOfLifeModel(input_map, output_map, model_type, model_path, time_step)
    game.run()
    return game.model.alive_counts, game.model.alive_grids

def plot_alive_counts(pcraster_alive_counts, backend_alive_counts, time_steps):
    plt.figure(figsize=(12, 8))
    plt.plot(range(time_steps + 1), pcraster_alive_counts, marker='s', linestyle='-', color='orange', label='PCRaster')
    plt.plot(range(time_steps + 1), backend_alive_counts, marker='x', linestyle='--', color='blue', label='ONNX ML Model')
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Alive Cells', fontsize=14)
    plt.title('Comparison of Alive Cells Over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def calculate_mse_and_acc(pcraster_grids, backend_grids):
    mse_list = []
    acc_list = []

    for pcraster_grid, backend_grid in zip(pcraster_grids, backend_grids):
        mse = np.mean((pcraster_grid - backend_grid) ** 2)
        acc = np.mean(pcraster_grid == backend_grid)
        mse_list.append(mse)
        acc_list.append(acc)
    
    return mse_list, acc_list

def plot_mse_and_acc(mse_list, acc_list, time_steps):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Time Step', fontsize=14)
    ax1.set_ylabel('MSE', fontsize=14, color='tab:blue')
    ax1.plot(range(time_steps + 1), mse_list, marker='o', linestyle='-', color='tab:blue', label='MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=14, color='tab:green')
    ax2.plot(range(time_steps + 1), acc_list, marker='s', linestyle='--', color='tab:green', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('MSE and Accuracy of ONNX Model Compared to PCRaster Model Over Time', fontsize=16)
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    plt.show()

def calculate_average_alive_cells(alive_counts):
    return np.mean(alive_counts)


