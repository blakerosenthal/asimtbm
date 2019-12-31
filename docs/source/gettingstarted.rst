
Getting Started
===============

asimtbm includes two examples for getting started.  The first example, ``example``, is an aggregate destination choice
model with doubly-constrained balancing.  The second example, ``example_balance``, is a doubly-constrained
matrix balancer.  In the first example, an OD matrix of trips is created using a customizable destination choice model
and then trip destinations are balanced to input destination control totals.  In the second example, an OD matrix of 
trips is input and then it is balanced to the origin and destination zone control totals.  

Installation
------------

* Install `Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__, which includes a number of required Python packages.
* Create and activate an Anaconda environment (i.e. a Python install just for this project).

::

  conda create -n asimtbmtest python=3.7
  activate asimtbmtest

* Get and install the asimtbm package from `GitHub <https://github.com/RSGInc/asimtbm>`_

::

  pip install https://github.com/RSGInc/asimtbm/zipball/master


Running the Model
-----------------

* Activate the conda Python environment

::

  activate asimtbmtest

* Change to the ``example`` or ``example_balance`` folder and then run the ``run_asimtbm.py`` program

::

  python run_asimtbm.py

* Check the outputs folder for results
