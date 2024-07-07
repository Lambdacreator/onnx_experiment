import numpy as np
import onnxruntime as ort
import onnx
import matplotlib.pyplot as plt
from matplotlib.pyplot import Button
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

def simulate_process(input_map, output_map, model_type, model_path, time_step=5):
    class GameOfLifeModel(DynamicFramework):
        def __init__(self):
            model = GameOfLife(input_map, output_map, model_path, model_type)
            DynamicFramework.__init__(self, model, time_step)
            self.model_type = model_type
            print(f"GameOfLifeModel initialized with model_type: {model_type}")
    
        def dynamic(self):
            self.model.dynamic()

    if not os.path.exists(input_map):
        raise FileNotFoundError(f"Input map file {input_map} not found.")
    
    if model_type in ['runtime', 'backend'] and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    print(f"Running Game of Life with model type: {model_type}")
    game = GameOfLifeModel()
    game.run()

    # Plotting the figures
    images = []
    aliveml = readmap(output_map)
    aliveml_np = pcr2numpy(aliveml, 2)
    images.append((0, aliveml_np))

    for i in range(1, time_step + 1):
        raster = readmap(f'result00.{i:03d}')
        raster_data = pcr2numpy(raster, 2)
        images.append((i, raster_data))

    # Plot the images with navigation
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(images[0][1])
    ax.set_title(f'{model_type} - Time Step 0')
    current_index = 0

    def update_plot(index):
        ax.clear()
        ax.imshow(images[index][1])
        ax.set_title(f'{model_type} - Time Step {images[index][0]}')
        fig.canvas.draw()

    def next(event):
        nonlocal current_index
        current_index = (current_index + 1) % len(images)
        update_plot(current_index)

    def prev(event):
        nonlocal current_index
        current_index = (current_index - 1) % len(images)
        update_plot(current_index)

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(prev)

    plt.show()

# Example call to the function
if __name__ == "__main__":
    simulate_process('alive.map', 'result', 'backend', 'game_of_life.onnx', 100)

