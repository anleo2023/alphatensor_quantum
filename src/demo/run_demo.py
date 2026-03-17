# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Demo for AlphaTensor-Quantum.

This demo showcases how to connect the components of AlphaTensor-Quantum with an
existing third-party library, MCTX (https://github.com/google-deepmind/mctx).
MCTX is a package for training and evaluating AlphaZero agents on a variety of
games.

Inspired by the MCTX demo at https://github.com/kenjyoung/mctx_learning_demo,
we use MCTX to build a simplified version of AlphaTensor-Quantum that can run on
a single machine (we strongly recommend access to a GPU to speed up the code).
Despite its simplicity, our demo is able to reproduce the following results of
the AlphaTensor-Quantum paper:
- Best reported T-count for three benchmark targets (Mod 5_4, Barenco Toff 3,
  and NC Toff 3) when running without gadgets (`use_gadgets=False`). This takes
  about 7800 iterations of the training loop on a Nvidia Quadro P1000 GPU.
- Best reported T-count for one benchmark target (Mod 5_4) when running with
  gadgets (`use_gadgets=True`). This takes about 450 iterations on the same GPU.

Our demo is intended to be a starting point for practitioners and researchers to
build on; it is by no means a complete implementation able to reproduce all the
results reported in the AlphaTensor-Quantum paper.

See the repository `README.md` for instructions on how to run the demo.
"""
from gc import enable
import os
import time
import numpy as np 


from absl import app
import jax
import jax.numpy as jnp

from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import demo_config
from alphatensor_quantum.src import tensors

#-----------------------------------------------------------------------------------
# interaction 
#-----------------------------------------------------------------------------------
def _prompt_yes_no(question: str) -> bool:
    """Prompts the user with a yes/no question and returns their response."""
    while True:
        response = input(f'{question} (y/n): ').strip().lower()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print('Invalid response. Please enter "y" or "n".')
 
def _load_tensor() -> str:
    print('\n Please provide the path to the custom tensor file (in .npy format).')
    while True:
        path = input('Path: ').strip()
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isfile(path):
            print(f'File not found at {path}. Please try again.')
            continue
        try:
            tensors.load_custom_tensors(path)
            print('Custom tensors loaded successfully.')
            return path
        except ValueError as e:
            print(f'Error loading tensors: {e}. Please try again.')
            print('  Please provide a valid 3D tensor file.')
 
def _prompt_use_gadgets() -> bool:
    """Ask whether to enable gadgetization."""
    print(
        '\nGadgetization replaces groups of T-gates with cheaper Toffoli gadgets,'
        '\nwhich can lower the effective T-count for some circuits.'
    )
    return _prompt_yes_no('Enable gadgetization?')
 
def _prompt_output_filename(output_dir: str) -> str:
    """Ask the user for the output filename."""
    while True:
        name = input('Enter a name for the output file (without extension): ').strip()
        if not name:
            print('  Name cannot be empty. Please try again.')
            continue
        full_path = f'{output_dir}/{name}.npy'
        if os.path.isfile(full_path):
            print(f'  File already exists at {full_path}.')
            overwrite = _prompt_yes_no('  Do you want to overwrite it?')
            if not overwrite:
                continue
        return name
 

#-----------------------------------------------------------------------------------
#Main 
#-----------------------------------------------------------------------------------



def main(_):
  ################################# user interaction to load custom tensor and set up the demo #################################
  print('=' * 60)
  print('       AlphaTensor-Quantum Demo')
  print('=' * 60)
        
  tensor_path = _load_tensor()
  circuit_name = os.path.splitext(os.path.basename(tensor_path))[0]
  output_dir = f'outputs/{circuit_name}'
  os.makedirs(output_dir, exist_ok=True)
  print(f'  Results will be saved to {output_dir}/')
  use_gadgets = _prompt_use_gadgets()
  output_filename = _prompt_output_filename(output_dir)
        
  print('\nStarting with:')
  print(f'  Gadgetization : {"enabled" if use_gadgets else "disabled"}')
  print()
  ########################################################
  
  # Set up the hyperparameters for the demo.
  config = demo_config.get_demo_config(
      use_gadgets=use_gadgets
  )
  exp_config = config.exp_config

  # Initialize the agent and the run state.
  agent = agent_lib.Agent(config)
  run_state = agent.init_run_state(jax.random.PRNGKey(2024))

  # Main loop.
  for step in range(
      0, exp_config.num_training_steps, exp_config.eval_frequency_steps
  ):
    time_start = time.time()
    run_state = agent.run_agent_env_interaction(step, run_state)
    time_taken = (time.time() - time_start) / exp_config.eval_frequency_steps
    # Keep track of the average return (for reporting purposes). We use a
    # debiased version of `avg_return` that only includes batch elements with at
    # least one completed episode.
    num_games = run_state.game_stats.num_games
    avg_return = run_state.game_stats.avg_return
    avg_return = jnp.sum(
        jnp.where(
            num_games > 0,
            avg_return / (1.0 - exp_config.avg_return_smoothing ** num_games),
            0.0
        ),
        axis=0
    ) / jnp.sum(num_games > 0, axis=0)
    print(
        f'Step: {step + exp_config.eval_frequency_steps} .. '
        f'Running Average Returns: {avg_return} .. '
        f'Time taken: {time_taken} seconds/step'
    )
    for t, target_circuit in enumerate(config.env_config.target_circuit_types):
      tcount = int(-run_state.game_stats.best_return[t])
      print(f'  Best T-count for {target_circuit.name.lower()}: {tcount}')
      
    best_num_moves = int(run_state.game_stats.best_num_moves[0])
    best_factors = np.array(run_state.game_stats.best_factors[0])
    best_factors = best_factors[-best_num_moves:]  # trim zero padding
    np.save(f'{output_dir}/{output_filename}.npy', best_factors)
    print(f'  Best T-count so far: {int(-run_state.game_stats.best_return[0])} — saved {best_num_moves} factors to {output_dir}/{output_filename}.npy')

if __name__ == '__main__':
  app.run(main)
