"""
ToRSh: PyTorch-compatible deep learning framework built in Rust

ToRSh provides a high-performance, memory-safe alternative to PyTorch
with full API compatibility and superior performance characteristics.
"""

from typing import Optional, Union, List, Tuple, Any
import os
import sys

# Import the Rust extension module
try:
    from . import _C
except ImportError:
    # Fallback for development
    import rstorch._C as _C

# Re-export core classes and functions
from ._C import (
    Tensor,
    dtype,
    device,
    TorshError,
    
    # Creation functions (all provided by the Rust extension)
    tensor,
    zeros,
    ones,
    randn,
    rand,
    arange,
    linspace,
    eye,
    empty,
    full,
    zeros_like,
    ones_like,
    empty_like,
    full_like,
    randn_like,
    rand_like,

    # Device and dtype constants
    # NOTE: uint16 is intentionally absent — torsh_core::DType has no U16
    # variant, so the extension does not register it (see FOLLOW-UP: add
    # DType::U16 to torsh-core to restore uint16 support).
    float32, float64,
    int8, int16, int32, int64,
    uint8, uint32, uint64,
    bool as bool_dtype,
    cpu,
    
    # Utility functions
    device_count,
    is_available,
    cuda_is_available,
    mps_is_available,
    get_device_name,
)

# Import submodules
from . import nn
from . import optim
from . import autograd
from . import distributed
from . import functional as F

# Version information
__version__ = _C.__version__

# Default tensor type
_default_dtype = float32
_default_device = cpu

def set_default_dtype(dtype: 'dtype') -> None:
    """Set the default floating point dtype."""
    global _default_dtype
    _default_dtype = dtype

def get_default_dtype() -> 'dtype':
    """Get the default floating point dtype."""
    return _default_dtype

def set_default_device(device: Union[str, 'device']) -> None:
    """Set the default device for new tensors."""
    global _default_device
    if isinstance(device, str):
        _default_device = _C.device(device)
    else:
        _default_device = device

def get_default_device() -> 'device':
    """Get the default device for new tensors."""
    return _default_device

# NOTE: The tensor-creation helpers (zeros_like, ones_like, empty, empty_like,
# full, full_like, randn_like, rand_like) are imported directly from the Rust
# extension above. The previous pure-Python shims defined here silently returned
# wrong results (e.g. `full` ignored `fill_value` and returned zeros), so they
# were removed in favour of the real `_C` implementations.

# Math operations
def add(input: Tensor, other: Union[Tensor, float], *, alpha: float = 1.0) -> Tensor:
    """Add tensors element-wise: ``input + alpha * other``."""
    if isinstance(other, (int, float)):
        return input + tensor(other * alpha, dtype=input.dtype, device=input.device)
    else:
        scaled = other if alpha == 1.0 else mul(other, alpha)
        return input + scaled

def sub(input: Tensor, other: Union[Tensor, float], *, alpha: float = 1.0) -> Tensor:
    """Subtract tensors element-wise: ``input - alpha * other``."""
    if isinstance(other, (int, float)):
        return input - tensor(other * alpha, dtype=input.dtype, device=input.device)
    else:
        scaled = other if alpha == 1.0 else mul(other, alpha)
        return input - scaled

def mul(input: Tensor, other: Union[Tensor, float]) -> Tensor:
    """Multiply tensors element-wise."""
    if isinstance(other, (int, float)):
        return input * tensor(other, dtype=input.dtype, device=input.device)
    else:
        return input * other

def div(input: Tensor, other: Union[Tensor, float]) -> Tensor:
    """Divide tensors element-wise."""
    if isinstance(other, (int, float)):
        return input / tensor(other, dtype=input.dtype, device=input.device)
    else:
        return input / other

def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Matrix multiplication."""
    return input @ other

def mm(input: Tensor, mat2: Tensor) -> Tensor:
    """Matrix multiplication (2D only)."""
    return input @ mat2

def bmm(input: Tensor, mat2: Tensor) -> Tensor:
    """Batch matrix multiplication."""
    return input @ mat2

# Shape operations
def reshape(input: Tensor, shape: List[int]) -> Tensor:
    """Reshape tensor."""
    return input.reshape(shape)

def view(input: Tensor, shape: List[int]) -> Tensor:
    """View tensor with new shape."""
    return input.view(shape)

def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Transpose dimensions."""
    return input.transpose(dim0, dim1)

def permute(input: Tensor, dims: List[int]) -> Tensor:
    """Permute dimensions."""
    return input.permute(dims)

def squeeze(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1."""
    return input.squeeze(dim)

def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1."""
    return input.unsqueeze(dim)

def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten tensor."""
    return input.flatten(start_dim, end_dim)

# Compatibility aliases
FloatTensor = Tensor
LongTensor = Tensor
IntTensor = Tensor
ByteTensor = Tensor
BoolTensor = Tensor

# Set module attributes for better IDE support
__all__ = [
    # Core classes
    'Tensor', 'dtype', 'device', 'TorshError',
    
    # Submodules
    'nn', 'optim', 'autograd', 'distributed', 'F',
    
    # Creation functions
    'tensor', 'zeros', 'ones', 'randn', 'rand', 'arange', 'linspace', 'eye',
    'zeros_like', 'ones_like', 'randn_like', 'rand_like',
    'empty', 'empty_like', 'full', 'full_like',
    
    # Math operations
    'add', 'sub', 'mul', 'div', 'matmul', 'mm', 'bmm',
    
    # Shape operations
    'reshape', 'view', 'transpose', 'permute', 'squeeze', 'unsqueeze', 'flatten',
    
    # Device and dtype constants
    'float32', 'float64', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint32', 'uint64', 'bool_dtype', 'cpu',
    
    # Utility functions
    'device_count', 'is_available', 'cuda_is_available', 'mps_is_available',
    'set_default_dtype', 'get_default_dtype', 'set_default_device', 'get_default_device',
    
    # Compatibility aliases
    'FloatTensor', 'LongTensor', 'IntTensor', 'ByteTensor', 'BoolTensor',
]