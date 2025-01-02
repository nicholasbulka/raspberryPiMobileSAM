from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class PhysicsMask:
    """
    Represents a mask with physical properties for motion interaction.
    Implements numpy's array interface for seamless integration with numpy operations.
    """
    mask: np.ndarray           # The actual mask data
    position: Tuple[int, int]  # Current (x, y) position of mask center
    dx: float = 0.0           # Horizontal velocity
    dy: float = 0.0           # Vertical velocity
    friction: float = 0.95    # Friction coefficient to gradually slow movement
    mass: float = 1.0        # Mass affects how much motion impacts the mask
    is_active: bool = True    # Whether the mask is still in play

    def __array__(self) -> np.ndarray:
        """
        Implementation of numpy's array interface.
        This allows PhysicsMask objects to be used directly in numpy operations.
        """
        return np.array(self.mask)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Support for numpy's universal functions (ufuncs).
        This enables operations like np.add, np.multiply, etc. to work with PhysicsMask objects.
        """
        # Convert all PhysicsMask inputs to their underlying arrays
        arrays = [(x.mask if isinstance(x, PhysicsMask) else x) for x in inputs]
        
        # Apply the ufunc to the arrays
        result = getattr(ufunc, method)(*arrays, **kwargs)
        
        # If the result should be a PhysicsMask, wrap it
        if method == '__call__' and isinstance(result, np.ndarray):
            return PhysicsMask(
                mask=result,
                position=self.position,
                dx=self.dx,
                dy=self.dy,
                friction=self.friction,
                mass=self.mass,
                is_active=self.is_active
            )
        return result

    def tolist(self) -> list:
        """
        Convert the mask to a Python list.
        This enables JSON serialization for web interface communication.
        """
        return self.mask.tolist()

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounding box of the mask (x1, y1, x2, y2)."""
        y_indices, x_indices = np.where(self.mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, 0, 0)
        return (
            int(np.min(x_indices)),
            int(np.min(y_indices)),
            int(np.max(x_indices)),
            int(np.max(y_indices))
        )
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate the center of the mask."""
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def copy(self) -> 'PhysicsMask':
        """
        Create a deep copy of the PhysicsMask.
        This ensures that modifying the copy doesn't affect the original.
        """
        return PhysicsMask(
            mask=self.mask.copy(),
            position=self.position,
            dx=self.dx,
            dy=self.dy,
            friction=self.friction,
            mass=self.mass,
            is_active=self.is_active
        )

    def __bool__(self) -> bool:
        """
        Enable truth value testing.
        A PhysicsMask is considered True if it has any True values in its mask.
        """
        return bool(np.any(self.mask))

    def __len__(self) -> int:
        """
        Return the number of True values in the mask.
        This is useful for checking if the mask is empty.
        """
        return int(np.sum(self.mask))

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the underlying mask array.
        This makes PhysicsMask objects more numpy-like.
        """
        return self.mask.shape

    @property
    def size(self) -> int:
        """
        Return the total size of the mask array.
        This makes PhysicsMask objects more numpy-like.
        """
        return self.mask.size

    def astype(self, dtype):
        """
        Convert the mask to a different data type, maintaining physics properties.
        
        Args:
            dtype: The target numpy dtype (e.g., np.uint8, np.float32)
            
        Returns:
            PhysicsMask: A new PhysicsMask with the mask data converted to the target type
        """
        new_mask = self.mask.astype(dtype)
        return PhysicsMask(
            mask=new_mask,
            position=self.position,
            dx=self.dx,
            dy=self.dy,
            friction=self.friction,
            mass=self.mass,
            is_active=self.is_active
        )

    def __getattr__(self, name):
        """
        Delegate unknown attribute access to the underlying numpy array.
        This allows PhysicsMask to support all numpy array methods transparently.
        
        Args:
            name: The name of the requested attribute
            
        Returns:
            The result of the operation, wrapped in PhysicsMask if appropriate
        """
        # Get the attribute from the underlying numpy array
        array_attr = getattr(self.mask, name)
        
        # If it's a method, wrap it to maintain PhysicsMask properties
        if callable(array_attr):
            def wrapped_method(*args, **kwargs):
                result = array_attr(*args, **kwargs)
                # If the result is a numpy array, wrap it in a PhysicsMask
                if isinstance(result, np.ndarray):
                    return PhysicsMask(
                        mask=result,
                        position=self.position,
                        dx=self.dx,
                        dy=self.dy,
                        friction=self.friction,
                        mass=self.mass,
                        is_active=self.is_active
                    )
                return result
            return wrapped_method
        return array_attr

    def __getitem__(self, key):
        """
        Support array indexing operations.
        This allows PhysicsMask to be indexed like a numpy array.
        
        Args:
            key: The index or slice to access
            
        Returns:
            The indexed data, wrapped in PhysicsMask if appropriate
        """
        result = self.mask[key]
        if isinstance(result, np.ndarray):
            return PhysicsMask(
                mask=result,
                position=self.position,
                dx=self.dx,
                dy=self.dy,
                friction=self.friction,
                mass=self.mass,
                is_active=self.is_active
            )
        return result