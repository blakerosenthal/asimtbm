import os
from activitysim.core import tracing


def setup_working_dir(example_name):

    example_dir = os.path.join(os.path.dirname(__file__), '../..', example_name)
    os.chdir(example_dir)

    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('log')
    tracing.delete_output_files('h5')
