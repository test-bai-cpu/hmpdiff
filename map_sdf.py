import numpy as np
from scipy import ndimage

class MapSDF:
    """Signed Distance Field from an occupancy grid map.
    
    Positive = distance to nearest obstacle (in free space)
    Negative = distance to nearest free space (inside obstacle)
    """
    
    def __init__(self, gray_map: np.ndarray, resolution: float,
                 origin_x: float, origin_y: float,
                 origin_at_bottom_left: bool = True,
                 free_pixel_threshold: int = 128):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_at_bottom_left = origin_at_bottom_left
        H, W = gray_map.shape
        self.H, self.W = H, W
        
        free_mask = gray_map >= free_pixel_threshold  # True = free space
        
        # Distance from each free cell to nearest obstacle
        dist_to_obstacle = ndimage.distance_transform_edt(free_mask)
        # Distance from each obstacle cell to nearest free space
        dist_to_free = ndimage.distance_transform_edt(~free_mask)
        
        # SDF: positive in free space, negative inside obstacles
        self._sdf_pixels = dist_to_obstacle - dist_to_free  # in pixels
        self._sdf_meters = self._sdf_pixels * resolution     # in meters
    
    @property
    def sdf_pixels(self) -> np.ndarray:
        """SDF in pixel units, shape (H, W)."""
        return self._sdf_pixels
    
    @property
    def sdf_meters(self) -> np.ndarray:
        """SDF in meters, shape (H, W)."""
        return self._sdf_meters
    
    def world_to_pixel(self, x: np.ndarray, y: np.ndarray):
        """Convert world coords to pixel (u, v)."""
        u = np.round((x - self.origin_x) / self.resolution).astype(int)
        v_f = (y - self.origin_y) / self.resolution
        if self.origin_at_bottom_left:
            v = np.round(self.H - 1 - v_f).astype(int)
        else:
            v = np.round(v_f).astype(int)
        return u, v
    
    def query_world(self, x, y, meters: bool = True) -> np.ndarray:
        """Query SDF at world coordinates. Out-of-bounds returns NaN."""
        # set meters to True to get SDF in meters, False for pixels
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        u, v = self.world_to_pixel(x, y)
        
        result = np.full(x.shape, np.nan, dtype=float)
        inside = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        
        sdf = self._sdf_meters if meters else self._sdf_pixels
        result[inside] = sdf[v[inside], u[inside]]
        return result
    
    def query_pixel(self, u, v, meters: bool = True) -> np.ndarray:
        # set meters to True to get SDF in meters, False for pixels
        """Query SDF at pixel coordinates. Out-of-bounds returns NaN."""
        u, v = np.asarray(u, dtype=int), np.asarray(v, dtype=int)
        result = np.full(u.shape, np.nan, dtype=float)
        inside = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        
        sdf = self._sdf_meters if meters else self._sdf_pixels
        result[inside] = sdf[v[inside], u[inside]]
        return result