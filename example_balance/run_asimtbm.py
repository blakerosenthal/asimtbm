import os

from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core.config import setting

# importing asimtbm also registers injectibles
import asimtbm
from asimtbm.test.utils import setup_working_dir

asimtbm.config_logger()

# Use zone data from main example
setup_working_dir('example_balance', inherit=True)

pipeline.run(models=setting('models'))
# pipeline.run(models=setting('models'), resume_after='destination_choice')

# tables will no longer be available after pipeline is closed
pipeline.close_pipeline()
