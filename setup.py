from setuptools import setup, find_packages, Extension

setup(
    name="rl_sandbox", 
    packages=find_packages(), 
    install_requires=["numpy==1.16.4", "tensorflow==2.7.2", "tqdm"]
)