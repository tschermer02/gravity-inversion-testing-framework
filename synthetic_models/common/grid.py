# A single compact density body.  
# Grid size: 64 × 64 × 24
# Body shape: rectangular compact block

from dataclasses import dataclass

# Documents the grid expected by the CNN.
# Gives the experiment a consistent coordinate system.
@dataclass(frozen=True)
class GridSpec:
    """Numerical and physical grid specifications for a single compact body."""
    nx: int = 64
    ny: int = 64
    nz: int = 24        

    x_min: float = 0.0
    x_max: float = 630.0

    y_min: float = 0.0
    y_max: float = 630.0

    z_min: float = 0.0
    z_max: float = 23.0
    
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)    
    
    @property
    def dy(self) -> float:      
        return (self.y_max - self.y_min) / (self.ny - 1)    
    
    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / (self.nz - 1)
    

        