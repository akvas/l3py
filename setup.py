import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='l3py',
    version='0.1',
    author='Andreas Kvas',
    description='A python package to convert potential coefficients to gridded mass anomalies',
    long_description=long_description,
    install_requires=['numpy', 'netcdf4'],
    packages=['l3py'],
    package_data={'l3py': ['data/DDK*_n2-120_n01Unchanged.npz', 'data/loadLoveNumbers_Gegout97.txt']}
)