import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='l3py',
    version='0.2',
    author='Andreas Kvas',
    description='A python package to convert potential coefficients to gridded mass anomalies',
    long_description=long_description,
    install_requires=['numpy', 'netcdf4', 'numpydoc'],
    packages=['l3py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={'l3py': ['data/ddk_normals.npz', 'data/loadLoveNumbers_Gegout97.txt']}
)
