from setuptools import setup, find_packages, Extension

setup(
    name="rl_sandbox", 
    packages=find_packages(), 
    install_requires=["numpy==1.22.0", "tensorflow==2.9.3", "tqdm"]
)