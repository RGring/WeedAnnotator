#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:/home/rog/repositories/WeedAnnotator"
export PYTHONPATH="${PYTHONPATH}:/home/rog/repositories/WeedAnnotator/weed_annotator"

venv/bin/python -m weed_annotator.full_pipeline.run_full_pipeline --config_folder "configs"
