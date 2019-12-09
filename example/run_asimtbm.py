from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core.config import setting

# importing asimtbm also registers injectibles
import asimtbm

asimtbm.config_logger()
tracing.delete_csv_files()

t0 = tracing.print_elapsed_time()

pipeline.run(models=setting('models'))
# pipeline.run(models=setting('models'), resume_after='destination_choice')

# tables will no longer be available after pipeline is closed
pipeline.close_pipeline()

tracing.print_elapsed_time("all models", t0)
