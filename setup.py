from setuptools import setup, find_packages

setup(
    name = 'simple_rl',
    
    version = '0.1.0',
    
    author = 'ChanJun',

    author_email = 'willy2space@gmail.com',

    description = 'Implementing reinforcement learning algorithms with simple code.',

    packages = find_packages(),

    install_requires = [
        'numpy==1.*',
        'torch==2.3.*',
        'matplotlib==3.8.*',
        'stable-baselines3==2.3.*'
    ]
)