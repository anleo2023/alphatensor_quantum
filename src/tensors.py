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

"""Tensor utilities for AlphaTensor-Quantum."""

import enum
import immutabledict

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


_CUSTOM_TENSORS = None

MAX_TENSOR_SIZE = 70  # the maximum allowed size for custom tensors, to prevent memory issues


_SMALL_TCOUNT_3 = np.array(
    [
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    ],
    dtype=np.int32,
)

_BARENCO_TOFF_3 = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_MOD_5_4 = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_NC_TOFF_3 = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

###################### To load custom tensors from .npy files ######################
def load_custom_tensors(path: str, max_size: int = MAX_TENSOR_SIZE) -> None:
    """loads a .npy file and sets it as the custom tensors 
    
    Args: 
        path: Path to the .npy file containing tensor 
        max_size : Maximum allowed size for the tensors
    Raises:
        ValueError : if the tensor is too large or not 3D
        
    """
    loaded = np.load(path)
    if loaded.ndim == 2:
        raise ValueError(
            f'Expected a 3D tensor, got a 2D array of shape {loaded.shape}.'
        )
    if loaded.ndim != 3:
        raise ValueError(
            f'Expected a 3D tensor, got shape {loaded.shape}.'
        )
    if not (loaded.shape[0] == loaded.shape[1] == loaded.shape[2]):
        raise ValueError(
            f'Tensor must be cubic (n x n x n), got shape {loaded.shape}.'
        )
    if loaded.shape[0] > max_size:
        raise ValueError(
            f'Tensor size {loaded.shape[0]} exceeds the maximum allowed size'
            f' of {max_size}.'
        )
    global _CUSTOM_TENSORS
    _CUSTOM_TENSORS = loaded.astype(np.int32)
#####################################################################################




class CircuitType(enum.Enum):
  """Types of circuits."""
  # Some circuits taken from the "Benchmarks" section of the paper.
  BARENCO_TOFF_3 = 1
  MOD_5_4 = 2
  NC_TOFF_3 = 3
  # A small 3-qubit circuit with optimal T-count of 3, useful for testing.
  SMALL_TCOUNT_3 = 4
  CUSTOM = 5  # custom tensors loaded  

  


_TENSORS_DICT = immutabledict.immutabledict({
    CircuitType.BARENCO_TOFF_3: _BARENCO_TOFF_3,
    CircuitType.MOD_5_4: _MOD_5_4,
    CircuitType.NC_TOFF_3: _NC_TOFF_3,
    CircuitType.SMALL_TCOUNT_3: _SMALL_TCOUNT_3,
})


def zero_pad_tensor(
    tensor: jt.Integer[jt.Array, 'size size size'],
    pad_to_size: int
) -> jt.Integer[jt.Array, '{pad_to_size} {pad_to_size} {pad_to_size}']:
  """Zero-pads the given tensor to the given size.

  Args:
    tensor: The tensor to pad.
    pad_to_size: The size to pad to. It must be at least as large as the tensor
      size.

  Returns:
    The padded tensor, such that the original tensor can be recovered by keeping
    the first `size` entries of each dimension.
  """
  size = tensor.shape[0]
  padding_width = pad_to_size - size
  return jnp.pad(tensor, (0, padding_width))



################### I changed this function to deal with custom tensors####################
def get_signature_tensor(
    circuit_type: CircuitType
) -> jt.Integer[jt.Array, 'size size size']:
  """Returns the signature tensor for the given quantum circuit.

  Args:
    circuit_type: The circuit type.

  Returns:
    The (symmetric) target signature tensor, with entries in {0, 1}.
  """
  if circuit_type == CircuitType.CUSTOM:
      if _CUSTOM_TENSORS is None:
          raise ValueError('Call load_custom_tensor(path) first.')
      return jnp.array(_CUSTOM_TENSORS)

  if circuit_type not in _TENSORS_DICT:
    raise ValueError(f'Unsupported circuit type: {circuit_type}')
  return jnp.array(_TENSORS_DICT[circuit_type])
######################################################################################