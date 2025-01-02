import numpy as np
import colorsys
import datetime
import random
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

# Global cache for colors
mask_colors: Dict[int, np.ndarray] = {}

@dataclass
class ColorScheme:
    """
    Color scheme for the application using the natural, earthy palette.
    All colors are stored in BGR format for OpenCV compatibility.
    Original RGB values are preserved in comments for reference.
    """
    # Tea Green: RGB(211, 250, 214) -> BGR(214, 250, 211)
    MAIN_COLOR_BGR: Tuple[int, int, int] = (214, 250, 211)
    
    # Tea Green 2: RGB(209, 239, 181) -> BGR(181, 239, 209)
    SECONDARY_COLOR_BGR: Tuple[int, int, int] = (181, 239, 209)
    
    # Vanilla: RGB(237, 235, 160) -> BGR(160, 235, 237)
    HIGHLIGHT_COLOR_BGR: Tuple[int, int, int] = (160, 235, 237)
    
    # Sage: RGB(195, 196, 141) -> BGR(141, 196, 195)
    ACCENT_COLOR_BGR: Tuple[int, int, int] = (141, 196, 195)
    
    # Moss Green: RGB(146, 140, 111) -> BGR(111, 140, 146)
    SHADOW_COLOR_BGR: Tuple[int, int, int] = (111, 140, 146)
    
    @classmethod
    def generate_from_base(cls, index: int) -> np.ndarray:
        """
        Generate a color variation based on the core color scheme.
        Uses the five base colors and applies subtle variations while
        maintaining the natural, earthy aesthetic.
        """
        # Core colors in BGR format
        core_colors = [
            cls.MAIN_COLOR_BGR,
            cls.SECONDARY_COLOR_BGR,
            cls.HIGHLIGHT_COLOR_BGR,
            cls.ACCENT_COLOR_BGR,
            cls.SHADOW_COLOR_BGR
        ]
        
        # Select base color using golden ratio distribution
        golden_ratio = 0.618033988749895
        color_index = int((index * golden_ratio) % len(core_colors))
        base_color = np.array(core_colors[color_index])
        
        # Apply subtle variations to maintain the natural look
        # Use smaller variation range (-10, 10) to keep colors consistent
        variation = np.random.uniform(-10, 10, 3)
        varied_color = np.clip(base_color + variation, 0, 255)
        
        return varied_color.astype(np.uint8)

# Create a global instance of the color scheme
color_scheme = ColorScheme()

def generate_color(index: int) -> np.ndarray:
    """
    Generate a color using the predefined natural color scheme.
    Returns colors in BGR format for OpenCV compatibility.
    """
    if index in mask_colors:
        return mask_colors[index]
    
    color = color_scheme.generate_from_base(index)
    mask_colors[index] = color
    return color

def generate_color_old(index: int) -> np.ndarray:
    """Main color generation function that cycles through different methods based on current minute."""
    if index in mask_colors:
        return mask_colors[index]
    
    # Get current minute and use it to select color generation method
    current_minute = datetime.datetime.now().minute
    method_index = current_minute % 10
    
    # List of color generation methods
    color_methods = [
        generate_golden_ratio_colors,
        generate_complementary_colors,
        generate_triadic_colors,
        generate_pastel_colors,
        generate_neon_colors,
        generate_earth_colors,
        generate_rainbow_colors,
        generate_analogous_colors,
        generate_monochromatic_colors,
        generate_jewel_colors
    ]
    
    # Generate color using selected method
    selected_method = color_methods[method_index]
    color = selected_method(index)
    
    # Cache and return the generated color
    mask_colors[index] = color
    return color

# The following color generation functions are kept for reference
# but are no longer used in the main generate_color function

def generate_golden_ratio_colors(index: int) -> np.ndarray:
    """Generate colors using golden ratio method for good distribution."""
    golden_ratio = 0.618033988749895
    hue = (index * golden_ratio) % 1.0
    saturation = np.random.uniform(0.7, 1.0)
    value = np.random.uniform(0.8, 1.0)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    variation = np.random.uniform(-20, 20, 3)
    rgb = np.clip(rgb + variation, 0, 255)
    
    return rgb.astype(np.uint8)

def generate_complementary_colors(index: int) -> np.ndarray:
    """Generate colors that are complementary to previous colors."""
    base_hue = (index * 0.1) % 1.0
    complement_hue = (base_hue + 0.5) % 1.0
    use_complement = index % 2 == 0
    
    hue = complement_hue if use_complement else base_hue
    saturation = np.random.uniform(0.8, 0.9)
    value = np.random.uniform(0.8, 1.0)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_triadic_colors(index: int) -> np.ndarray:
    """Generate colors using triadic color harmony."""
    base_hue = (index * 0.15) % 1.0
    triad_offset = index % 3 * 0.333
    hue = (base_hue + triad_offset) % 1.0
    saturation = np.random.uniform(0.7, 1.0)
    value = np.random.uniform(0.9, 1.0)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_pastel_colors(index: int) -> np.ndarray:
    """Generate soft pastel colors."""
    hue = (index * 0.23) % 1.0
    saturation = np.random.uniform(0.3, 0.5)
    value = np.random.uniform(0.9, 1.0)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_neon_colors(index: int) -> np.ndarray:
    """Generate bright neon colors."""
    hue = (index * 0.27) % 1.0
    saturation = np.random.uniform(0.8, 1.0)
    value = 1.0
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    variation = np.random.uniform(-10, 10, 3)
    rgb = np.clip(rgb + variation, 0, 255)
    
    return rgb.astype(np.uint8)

def generate_earth_colors(index: int) -> np.ndarray:
    """Generate natural earth-tone colors."""
    earth_hues = [0.08, 0.11, 0.15, 0.25, 0.35]  # Browns, greens, etc.
    hue = earth_hues[index % len(earth_hues)]
    saturation = np.random.uniform(0.4, 0.7)
    value = np.random.uniform(0.5, 0.8)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_rainbow_colors(index: int) -> np.ndarray:
    """Generate colors cycling through rainbow spectrum."""
    hue = (index * 0.1) % 1.0
    saturation = 0.9
    value = 0.95
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_analogous_colors(index: int) -> np.ndarray:
    """Generate colors that are analogous (adjacent on color wheel)."""
    base_hue = (index * 0.2) % 1.0
    hue_offset = (index % 3 - 1) * 0.05
    hue = (base_hue + hue_offset) % 1.0
    saturation = np.random.uniform(0.7, 0.9)
    value = np.random.uniform(0.8, 1.0)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_monochromatic_colors(index: int) -> np.ndarray:
    """Generate variations of a single color."""
    base_hue = (index * 0.35) % 1.0
    saturation = 0.7 + (index % 3) * 0.1
    value = 0.7 + (index % 4) * 0.075
    
    rgb = np.array(colorsys.hsv_to_rgb(base_hue, saturation, value)) * 255
    return rgb.astype(np.uint8)

def generate_jewel_colors(index: int) -> np.ndarray:
    """Generate rich, deep jewel-tone colors."""
    jewel_hues = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]  # Ruby, emerald, sapphire, etc.
    hue = jewel_hues[index % len(jewel_hues)]
    saturation = np.random.uniform(0.8, 1.0)
    value = np.random.uniform(0.5, 0.7)
    
    rgb = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return rgb.astype(np.uint8)