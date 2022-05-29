from setuptools import setup

setup(
    name='lightFM_quickstart',
    version='0.1.0',
    packages=['lightFM_quickstart'],
    url='',
    license='Apache 2.0',
    author='Griffin Barich',
    author_email='griffin.barich@gmail.com',
    description='Quick Setup for a LightFM model',
    install_requires=['pandas>=1.3.5', 'lightfm>=1.16', 'tqdm>=4.64.0'],
    include_package_data=True,
    package_data={'': ['data/example_data.csv.gz']}
)
