import numpy as np
import gymnasium as gym

class DroneEnvironment(gym.Env):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.state_size = 10  # State: [drone_x, drone_y, parcel_x, parcel_y, dest_x, dest_y, has_parcel, obstacle_count, velocity_x, velocity_y]
        self.action_size = 4  # Actions: 0=up, 1=down, 2=left, 3=right
        
        self.action_space = gym.spaces.Discrete(self.action_size)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )
        
        
        self.drone_pos = np.array([0, 0], dtype=float)
        self.parcel_pos = np.array([5, 5], dtype=float)
        self.destination_pos = np.array([9, 9], dtype=float)
        self.has_parcel = False
        self.obstacles = self._generate_obstacles()
        self.steps = 0
        self.max_steps = 200

    def _generate_obstacles(self):
        obstacles = set()
        for _ in range(5):
            x, y = np.random.randint(0, self.grid_size, 2)
            obstacles.add((x, y))
        return obstacles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([0, 0], dtype=float)
        self.has_parcel = False
        self.steps = 0
        obs = self._get_state()
        info = {}
        return obs, info

    def _get_state(self):
        state = np.array([
            self.drone_pos[0],
            self.drone_pos[1],
            self.parcel_pos[0],
            self.parcel_pos[1],
            self.destination_pos[0],
            self.destination_pos[1],
            float(self.has_parcel),
            len(self.obstacles),
            0.0,  # velocity_x
            0.0   # velocity_y
        ], dtype=np.float32)
        return state

    def step(self, action):
        self.steps += 1
        prev_pos = self.drone_pos.copy()
        
        # Execute action
        if action == 0:  # up
            self.drone_pos[1] += 1
        elif action == 1:  # down
            self.drone_pos[1] -= 1
        elif action == 2:  # left
            self.drone_pos[0] -= 1
        elif action == 3:  # right
            self.drone_pos[0] += 1
        
        # Clamp to grid
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size - 1)
        
        reward = -0.1  # small step penalty
        terminated = False
        truncated = False
        info = {}
        
        # Check collision with obstacles (unsafe RL – penalize heavily)
        if tuple(self.drone_pos.astype(int)) in self.obstacles:
            reward = -50
            terminated = True
            return self._get_state(), reward, terminated, False, info
        
        # Pickup parcel
        if not self.has_parcel and np.allclose(self.drone_pos, self.parcel_pos, atol=0.5):
            self.has_parcel = True
            reward += 10
        
        # Deliver parcel to destination (goal)
        if self.has_parcel and np.allclose(self.drone_pos, self.destination_pos, atol=0.5):
            reward += 100
            terminated = True
            return self._get_state(), reward, terminated, False, info
        
        # Small reward for moving closer to parcel/destination
        if not self.has_parcel:
            dist_to_parcel = np.linalg.norm(self.drone_pos - self.parcel_pos)
            reward += max(0, 0.1 - dist_to_parcel * 0.01)
        else:
            dist_to_dest = np.linalg.norm(self.drone_pos - self.destination_pos)
            reward += max(0, 0.1 - dist_to_dest * 0.01)
        
        # Timeout penalty
        if self.steps >= self.max_steps:
            truncated = True
            reward -= 10
        
        return self._get_state(), reward, terminated, truncated, info
        
    def render(self, mode="human"):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for o_x, o_y in self.obstacles:
            grid[o_y][o_x] = "X"
        p_x, p_y = int(self.parcel_pos[0]), int(self.parcel_pos[1])
        d_x, d_y = int(self.destination_pos[0]), int(self.destination_pos[1])
        drone_x, drone_y = int(self.drone_pos[0]), int(self.drone_pos[1])
        grid[p_y][p_x] = "P" if not self.has_parcel else " "
        grid[d_y][d_x] = "G"
        symbol = "D" if self.has_parcel else "d"
        grid[drone_y][drone_x] = symbol
        for row in reversed(grid):  # Print with y=0 at bottom
            print(" ".join(row))
        print()
    
    def close(self):
        pass
