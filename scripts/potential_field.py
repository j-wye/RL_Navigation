import numpy as np

class PotentialFieldForce:
    def __init__(self, goal, pose, voxel_data, COLLISION_THRESHOLD):
        self.goal = goal
        self.pose = pose[:2]
        self.yaw = pose[2]
        self.voxel_data = voxel_data
        self.kappa_a = 1.0
        self.kappa_b = 10.0
        self.kappa_r = 100.0
        self.rho = COLLISION_THRESHOLD * 2.5
    
    def attractive_force(self):
        remain_dist = np.linalg.norm(self.goal - self.pose)
        if remain_dist <= self.rho:
            return self.kappa_a * (self.goal - self.pose)
        else:
            return self.kappa_b * (self.goal - self.pose) / remain_dist

    def repulsive_force(self):
        obj_local = np.stack([self.voxel_data[:, :, 0].ravel(),
                              self.voxel_data[:, :, 1].ravel()], axis=1)
        
        
        R = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                      [np.sin(self.yaw), np.cos(self.yaw)]])
        t = self.pose.reshape(2, 1)
        
        obj_global = (R @ obj_local.T + t).T
        dists = np.linalg.norm(obj_global - self.pose, axis=1)
        
        idx = np.argmin(dists)
        d_min = dists[idx]
        closest_obj = obj_global[idx]
        
        delta = self.pose - closest_obj
        
        if 0 <= d_min <= self.rho:
            direction = delta / d_min
            magnitude = self.kappa_r * (1 / d_min - 1 / self.rho) * (1 / d_min**2)
            return direction * magnitude
        else:
            return np.zeros(2, dtype=np.float32)

    def compute_force(self):
        F = self.attractive_force() + self.repulsive_force()
        yaw = np.arctan2(F[1], F[0])
        yaw_desired = (yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        return float(yaw_desired)

class PotentialFieldEnergy:
    def __init__(self, goal, pose, voxel_data, COLLISION_THRESHOLD):
        self.goal = goal
        self.pose = pose[:2]
        self.yaw = pose[2]
        self.voxel_data = voxel_data
        self.kappa_a = 1.0
        self.kappa_b = 2.0
        self.kappa_r = 100.0
        self.rho = COLLISION_THRESHOLD * 2.5
    
    def attractive_energy(self):
        remain_dist = np.linalg.norm(self.goal - self.pose)
        if 0 <= remain_dist <= self.rho:
            return 0.5 * self.kappa_a * (remain_dist**2)
        else:
            return self.kappa_b * remain_dist
    
    def repulsive_energy(self):
        obj_local = np.stack([self.voxel_data[:, :, 0].ravel(),
                              self.voxel_data[:, :, 1].ravel()], axis=1)
        
        R = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                      [np.sin(self.yaw), np.cos(self.yaw)]])
        t = self.pose.reshape(2, 1)
        
        obj_global = (R @ obj_local.T + t).T
        
        dists = np.linalg.norm(obj_global - self.pose, axis=1)
        d_min = np.min(dists) if len(dists) > 0 else float('inf')
        
        if 0 <= d_min <= self.rho:
            return 0.5 * self.kappa_r * ((1 / d_min - 1 / self.rho)**2)
        else:
            return 0.0
    
    def compute_energy(self):
        return float(self.attractive_energy() + self.repulsive_energy())