# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Logging functionality for buddy-based experiments."""

import os
from typing import Any, Mapping

from bsuite import environments
from bsuite import sweep
from bsuite.logging import base
from bsuite.utils import wrappers

import dm_env
import experiment_buddy

SAFE_SEPARATOR = '-'
INITIAL_SEPARATOR = '_-_'
BSUITE_PREFIX = 'bsuite_id' + INITIAL_SEPARATOR


def wrap_environment(env: environments.Environment,
                     bsuite_id: str,
                     log_by_step: bool = False) -> dm_env.Environment:
  """Returns a wrapped environment that logs using CSV."""
  logger = Logger(bsuite_id)
  return wrappers.Logging(env, logger, log_by_step=log_by_step)


class Logger(base.Logger):
  """Saves data to a CSV file via Pandas.

  In this simplified logger, each bsuite_id logs to a unique CSV index by
  bsuite_id. These are saved to a single results_dir by experiment.
  We strongly suggest that you use a *fresh* folder for each bsuite run.

  The write method rewrites the entire CSV file on each call. This is not
  intended to be an optimized example. However, writes are infrequent due to
  bsuite's logarithmically-spaced logging.

  This logger, along with the corresponding load functionality, serves as a
  simple, minimal example for users who need to implement logging to a different
  storage system.
  """

  def __init__(self,
               bsuite_id: str):
    # The default '/' symbol is dangerous for file systems!
    safe_bsuite_id = bsuite_id.replace(sweep.SEPARATOR, SAFE_SEPARATOR)
    filename = f'{BSUITE_PREFIX}{safe_bsuite_id}.csv'
    experiment_buddy.register({"dummy":"dummy"})
    self.writer = experiment_buddy.deploy(host="mila", proc_num=4)

  def write(self, data: Mapping[str, Any]):
    for k, v in data.items():
      self.writer.add_scalar(k, v, global_step=data['steps'])