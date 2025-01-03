from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import json
import cv2

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
    
    # Physics properties
    acceleration: Tuple[float, float] = (0.0, 0.0)  # Current acceleration
    force: Tuple[float, float] = (0.0, 0.0)        # Applied force
    rotation: float = 0.0                          # Rotation angle in degrees
    angular_velocity: float = 0.0                  # Rotational velocity (degrees/sec)
    angular_momentum: float = 0.0                  # Angular momentum
    moment_of_inertia: float = 1.0                # Moment of inertia
    torque: float = 0.0                           # Applied torque
    scale: float = 1.0                            # Scale factor
    
    # Decay properties
    lifetime: float = 100.0   # Starting lifetime value
    decay_rate: float = 1.0   # How fast the mask fades
    opacity: float = 1.0      # Current opacity
    
    # Visual content
    pixel_content: Optional[np.ndarray] = None  # Original pixels from frame
    
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
        
        # Calculate center of mass and moment of inertia
        self.update_mass_properties()
    
    def update_mass_properties(self):
        """Calculate center of mass and moment of inertia based on mask shape."""
        y_indices, x_indices = np.where(self.mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return
        
        # Calculate center of mass
        self.com_x = np.mean(x_indices)
        self.com_y = np.mean(y_indices)
        
        # Calculate moment of inertia around center of mass
        # For each pixel, I = mass * r^2
        dx = x_indices - self.com_x
        dy = y_indices - self.com_y
        r_squared = dx * dx + dy * dy
        self.moment_of_inertia = np.sum(r_squared) * self.mass / len(x_indices)
    
    def apply_torque(self, torque: float, dt: float):
        """Apply torque to cause rotation."""
        # Update angular momentum (L = L₀ + τΔt)
        self.angular_momentum += torque * dt
        
        # Update angular velocity (ω = L/I)
        self.angular_velocity = self.angular_momentum / self.moment_of_inertia
        
        # Update rotation angle (θ = θ₀ + ωΔt)
        self.rotation += self.angular_velocity * dt
        
        # Normalize rotation to 0-360 degrees
        self.rotation = self.rotation % 360
    
    def rotate_content(self):
        """Rotate pixel content around center of mass."""
        if self.pixel_content is None or self.rotation == 0:
            return
        
        # Get rotation matrix
        center = (self.com_x, self.com_y)
        M = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
        
        # Rotate pixel content
        rotated = cv2.warpAffine(
            self.pixel_content,
            M,
            (self.pixel_content.shape[1], self.pixel_content.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Rotate mask
        rotated_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            M,
            (self.mask.shape[1], self.mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT
        ).astype(bool)
        
        self.mask = rotated_mask
        self.pixel_content = rotated
    
    def update(self, dt: float) -> None:
        """
        Update mask state including decay and rotation.
        
        Args:
            dt: Time step in seconds
        """
        # Update lifetime and opacity
        self.lifetime -= self.decay_rate * dt
        self.opacity = max(0.0, self.lifetime / 100.0)
        
        # Deactivate if completely faded
        if self.lifetime <= 0 or self.opacity <= 0:
            self.is_active = False
            return
        
        # Apply angular friction
        angular_friction = 0.98  # Slightly less friction for rotation
        self.angular_velocity *= angular_friction
        
        # Update rotation
        if abs(self.angular_velocity) > 0.1:  # Minimum angular velocity threshold
            self.rotation += self.angular_velocity * dt
            self.rotation = self.rotation % 360
            self.rotate_content()
        
        # Update velocity based on decay
        velocity_scale = self.opacity  # Slow down as it fades
        self.dx *= velocity_scale
        self.dy *= velocity_scale
    
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
                angular_momentum=self.angular_momentum,
                moment_of_inertia=self.moment_of_inertia,
                scale=self.scale,
                lifetime=self.lifetime,
                decay_rate=self.decay_rate,
                opacity=self.opacity,
                pixel_content=self.pixel_content.copy() if self.pixel_content is not None else None
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
            'angular_velocity': float(self.angular_velocity),
            'scale': float(self.scale),
            'is_active': self.is_active,
            'bounds': self.bounds,
            'opacity': float(self.opacity)
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
        return (int(self.com_x), int(self.com_y))
    
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
        """Calculate total kinetic energy (translational + rotational)."""
        translational = 0.5 * self.mass * (self.dx * self.dx + self.dy * self.dy)
        rotational = 0.5 * self.moment_of_inertia * (np.radians(self.angular_velocity) ** 2)
        return translational + rotational
    
    def apply_force(self, fx: float, fy: float, point: Optional[Tuple[float, float]] = None) -> None:
        """
        Apply a force to the mask, updating both linear and angular motion.
        
        Args:
            fx: Force in x direction
            fy: Force in y direction
            point: Point of application (if None, applies to center of mass)
        """
        # Update linear motion (F = ma)
        self.acceleration = (fx / self.mass, fy / self.mass)
        self.dx += self.acceleration[0]
        self.dy += self.acceleration[1]
        
        # Calculate torque if force is not applied at center of mass
        if point is not None:
            # Vector from COM to application point
            rx = point[0] - self.com_x
            ry = point[1] - self.com_y
            
            # Torque = r × F
            torque = rx * fy - ry * fx
            self.torque = torque
            
            # Update angular motion
            self.angular_momentum += torque
            self.angular_velocity = self.angular_momentum / self.moment_of_inertia
    
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
        
        # Update rotation
        if abs(self.angular_velocity) > 0.1:  # Minimum threshold
            self.rotation += self.angular_velocity * dt
            self.rotation = self.rotation % 360
    
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
            angular_momentum=self.angular_momentum,
            moment_of_inertia=self.moment_of_inertia,
            scale=self.scale,
            lifetime=self.lifetime,
            decay_rate=self.decay_rate,
            opacity=self.opacity,
            pixel_content=self.pixel_content.copy() if self.pixel_content is not None else None
        )