from setuptools import setup, find_packages
from os import path
import re

name = 'asimtbm'
asim = {'__name__': name}
with open(path.join(name, '__init__.py')) as f:
    info = re.search(r'__.*', f.read(), re.S)
    exec(info[0], asim)

setup(
    name=name,
    version=asim['__version__'],
    description=asim['__doc__'],
    author=asim['__author__'],
    author_email='blake.rosenthal@rsginc.com',
    license=asim['__license__'],
    url='https://github.com/RSGInc/asimtbm',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(exclude=['*.tests']),
    include_package_data=True,
    python_requires=">=3.5.3",
    install_requires=[
        'numpy >= 1.16.1',
        'openmatrix >= 0.3.4.1',
        'pandas >= 0.24.1',
        'activitysim >= 0.9.1',
        'ipfn >= 1.3.0'
    ]
)
