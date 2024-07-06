from setuptools import setup, find_packages

setup(
    name='game_of_life',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'onnx',
        'onnxruntime',
        'matplotlib',
        # 'pcraster',  # Removed this since it cannot be installed via pip
    ],
    entry_points={
        'console_scripts': [
            'run_game_of_life=game_of_life.game_of_life:run_game_of_life',
        ],
    },
)
