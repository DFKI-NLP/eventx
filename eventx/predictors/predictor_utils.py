# Code copied from:
# https://github.com/ChristophAlt/daystream/blob/master/daystream/predictors/predictor_utils.py

from typing import Optional

from os.path import join
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive


def load_predictor(model_dir: str,
                   predictor_name: str,
                   cuda_device: int = -1,
                   archive_filename: str = "model.tar.gz",
                   weights_file: Optional[str] = None,
                   overrides="") -> Predictor:
    archive_path = join(model_dir, archive_filename)
    archive = load_archive(archive_path, cuda_device, overrides, weights_file)
    return Predictor.from_archive(archive, predictor_name)
