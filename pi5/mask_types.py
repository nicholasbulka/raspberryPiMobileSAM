from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import json

@dataclass
class PhysicsMask:
    """
    Enhanced physics-enabled mask with improved properties and methods.
    Implements numpy's array interface for seamless integration with numpy operations.
    """
    mask: np.ndarray           # The actual mask data
    position: Tuple[int, int]  # Current (x, y) position of mask center
    dx: float = 0.0           # Horizontal velocity
    dy: float = 0.0           # Vertical velocity
    friction: float = 0.95    # Friction coefficient (lower = more friction)
    mass: float = 1.0         # Mass affects force response
    is_active: bool = True    # Whether mask is still in play
    
    # New physics properties
    acceleration: Tuple[float, float] = (0.0, 0.0)  # Current acceleration
    force: Tuple[float, float] = (0.0, 0.0)        # Applied force
    rotation: float = 0.0                          # Rotation angle in degrees
    angular_velocity: float = 0.0                  # Rotational velocity
    scale: float = 1.0                            # Scale factor
    
    def __post_init__(self):
        """Validate and initialize additional properties after creation."""
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        
        if self.mask.dtype != bool:
            self.mask = self.mask.astype(bool)
            
        # Initialize physics state
        self.last_position = self.position
        self.last_update_time = None
        self.collision_count = 0
        self.stable_count = 0
    
    def __array__(self) -> np.ndarray:
        """Implementation of numpy's array interface."""
        return np.array(self.mask)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support for numpy's universal functions (ufuncs)."""
        # Convert PhysicsMask inputs to arrays
        arrays = [(x.mask if isinstance(x, PhysicsMask) else x) for x in inputs]
        
        # Apply the ufunc
        result = getattr(ufunc, method)(*arrays, **kwargs)
        
        # Return new PhysicsMask if appropriate
        if method == '__call__' and isinstance(result, np.ndarray):
            return PhysicsMask(
                mask=result,
                position=self.position,
                dx=self.dx,
                dy=self.dy,
                friction=self.friction,
                mass=self.mass,
                is_active=self.is_active,
                rotation=self.rotation,
                angular_velocity=self.angular_velocity,
                scale=self.scale
            )
        return result
    
    def tolist(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary with enhanced properties.
        Used for web interface communication.
        """
        return {
            'mask': self.mask.tolist(),
            'position': self.position,
            'velocity': (float(self.dx), float(self.dy)),
            'rotation': float(self.rotation),
            'scale': float(self.scale),
            'is_active': self.is_active,
            'bounds': self.bounds
        }
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounding box (x1, y1, x2, y2) with rotation support."""
        if not np.any(self.mask):
            return (0, 0, 0, 0)
            
        y_indices, x_indices = np.where(self.mask)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, 0, 0)
            
        # Account for rotation if present
        if self.rotation != 0:
            # Convert to center-relative coordinates
            center_x = np.mean(x_indices)
            center_y = np.mean(y_indices)
            rel_x = x_indices - center_x
            rel_y = y_indices - center_y
            
            # Rotate points
            angle_rad = np.radians(self.rotation)
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)
            
            rot_x = rel_x * cos_theta - rel_y * sin_theta + center_x
            rot_y = rel_x * sin_theta + rel_y * cos_theta + center_y
            
            x_indices = rot_x.astype(int)
            y_indices = rot_y.astype(int)
        
        return (
            int(np.min(x_indices)),
            int(np.min(y_indices)),
            int(np.max(x_indices)),
            int(np.max(y_indices))
        )
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate the center point of the mask."""
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity vector."""
        return (self.dx, self.dy)
    
    @property
    def speed(self) -> float:
        """Calculate current speed (magnitude of velocity)."""
        return np.sqrt(self.dx * self.dx + self.dy * self.dy)
    
    @property
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy of the mask."""
        return 0.5 * self.mass * (self.dx * self.dx + self.dy * self.dy)
    
    def apply_force(self, fx: float, fy: float) -> None:
        """
        Apply a force to the mask, updating acceleration and velocity.
        
        Args:
            fx: Force in x direction
            fy: Force in y direction
        """
        # Update acceleration (F = ma)
        self.acceleration = (fx / self.mass, fy / self.mass)
        
        # Update velocity (v = v0 + at)
        self.dx += self.acceleration[0]
        self.dy += self.acceleration[1]
    
    def update_position(self, dt: float) -> None:
        """
        Update position based on velocity and time step.
        
        Args:
            dt: Time step in seconds
        """
        # Update position (x = x0 + vt)
        new_x = self.position[0] + self.dx * dt
        new_y = self.position[1] + self.dy * dt
        self.position = (int(new_x), int(new_y))
    
    def handle_collision(self, other: 'PhysicsMask') -> None:
        """
        Handle collision with another mask using elastic collision physics.
        
        Args:
            other: The other PhysicsMask involved in collision
        """
        # Calculate collision normal
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            return
            
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        rvx = self.dx - other.dx
        rvy = self.dy - other.dy
        
        # Relative velocity along normal
        velAlongNormal = rvx * nx + rvy * ny
        
        # Don't resolve if objects are separating
        if velAlongNormal > 0:
            return
            
        # Coefficient of restitution (bounciness)
        restitution = 0.8
        
        # Collision impulse
        j = -(1 + restitution) * velAlongNormal
        j /= 1/self.mass + 1/other.mass
        
        # Apply impulse
        self.dx += (j * nx) / self.mass
        self.dy += (j * ny) / self.mass
        other.dx -= (j * nx) / other.mass
        other.dy -= (j * ny) / other.mass
    
    def copy(self) -> 'PhysicsMask':
        """Create a deep copy of the PhysicsMask."""
        return PhysicsMask(
            mask=self.mask.copy(),
            position=self.position,
            dx=self.dx,
            dy=self.dy,
            friction=self.friction,
            mass=self.mass,
            is_active=self.is_active,
            rotation=self.rotation,
            angular_velocity=self.angular_velocity,
            scale=self.scale
        )
    
    def __bool__(self) -> bool:
        """Enable truth value testing."""
        return bool(np.any(self.mask))
    
    def __len__(self) -> int:
        """Return the number of True values in the mask."""
        return int(np.sum(self.mask))
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying mask array."""
        return self.mask.shape
    
    @property
    def size(self) -> int:
        """Return the total size of the mask array."""
        return self.mask.size
    
    def astype(self, dtype) -> 'PhysicsMask':
        """Convert the mask to a different data type."""
        return PhysicsMask(
            mask=self.mask.astype(dtype),
            position=self.position,
            dx=self.dx,
            dy=self.dy,
            friction=self.friction,
            mass=self.mass,
            is_active=self.is_active,
            rotation=self.rotation,
            angular_velocity=self.angular_velocity,
            scale=self.scale
        )