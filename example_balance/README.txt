About:
  - asimtbm uses the activitysim pipeline framework to run steps in sequence. More information on
    activitysim can be found at https://activitysim.github.io.

Setup:
  - three folders are required:
    - output, an empty directory that asimtbm will write into.
    - data, a directory containing zone data, skims, and expressions. Row numbers will be used for
      zone data ids unless a 'zone' column is provided. Zones ids must be consistent between all
      zone files and skims.
    - configs, a directory containing yaml files that describe how to use the files in the data
      folder.

Run:
  - run_asimtbm.py provides a basic outline for running the example.
  - The main step is `pipeline.run(models=setting('models'))` which uses the activitysim pipeline
    to run each of the models specified in settings.yaml.
  - outputs can be found in the output directory.
