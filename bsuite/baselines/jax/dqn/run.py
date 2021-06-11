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
"""Run a Dqn agent instance (using JAX) on a bsuite experiment."""
import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.jax import dqn


FLAGS = {
  'bsuite_id': 'catch/0',
  'save_path': '/tmp/bsuite',
  'logging_mode': 'buddy',
  'overwrite': True,
  'num_episodes': None,
  'verbose': True,
}


def run(bsuite_id: str) -> str:
  """Runs a DQN agent on a given bsuite environment, logging to CSV."""

  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS['save_path'],
      logging_mode=FLAGS['logging_mode'],
      overwrite=FLAGS['overwrite'],
      config=FLAGS
  )

  agent = dqn.default_agent(env.observation_spec(), env.action_spec())

  num_episodes = FLAGS['num_episodes'] or getattr(env, 'bsuite_num_episodes')
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS['verbose'])

  return bsuite_id


def main():
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  bsuite_id = FLAGS['bsuite_id']

  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)
  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
  main()
