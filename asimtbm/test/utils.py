import os
from activitysim.core import tracing
from activitysim.core import inject


def example_dir(example_name):
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '../..',
                                        example_name))


def setup_working_dir(example_name, inherit=False):

    os.chdir(example_dir(example_name))

    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('log')
    tracing.delete_output_files('h5')

    if inherit:
        data_dir = inject.get_injectable('data_dir')
        example_data_dir = os.path.join(example_dir('example'), 'data')
        inject.add_injectable('data_dir', [data_dir, example_data_dir], cache=True)
