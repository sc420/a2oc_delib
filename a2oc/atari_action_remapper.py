import gym
import numpy as np


class AtariActionRemapperWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AtariActionRemapperWrapper, self).__init__(env)

        # Get action remap
        self.action_remap = self._get_action_remap(env)

        # Use new action space
        self.action_space = gym.spaces.Discrete(len(self.action_remap))

        # Inherit properties from gym

    def step(self, action):
        # Map the new action to old action
        old_action = self.action_remap[action]

        # Interact with old action
        obs, rew, done, info = self.env.step(old_action)

        return obs, rew, done, info

    def _get_action_remap(self, env):
        # Get environment ID
        env_id = env.spec.id

        # Determine environment ID
        if env_id == 'AsteroidsNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # FIRE
                2: 2,  # UP
                3: 3,  # RIGHT
                4: 4,  # LEFT
                5: 5,  # DOWN
                6: 6,  # UPRIGHT
                7: 7,  # UPLEFT
            }
        elif env_id == 'BeamRiderNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # FIRE
                2: 3,  # RIGHT
                3: 4,  # LEFT
            }
        elif env_id == 'BowlingNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # FIRE
                2: 2,  # UP
                3: 3,  # DOWN
            }
        elif env_id == 'BreakoutNoFrameskip-v4':
            return self._identity_action_remap(env)
        elif env_id == 'EnduroNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # FIRE (Accelerate)
                2: 2,  # RIGHT
                3: 3,  # LEFT
                4: 4,  # DOWN (Decelerate)
            }
        elif env_id == 'FreewayNoFrameskip-v4':
            return self._identity_action_remap(env)
        elif env_id == 'KungFuMasterNoFrameskip-v4':
            return self._identity_action_remap(env)
        elif env_id == 'MsPacmanNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # UP
                2: 2,  # RIGHT
                3: 3,  # LEFT
                4: 4,  # DOWN
            }
        elif env_id == 'PongNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 2,  # RIGHT
                2: 3,  # LEFT
            }
        elif env_id == 'QbertNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 2,  # UP
                2: 3,  # RIGHT
                3: 4,  # LEFT
                4: 5,  # DOWN
            }
        elif env_id == 'SeaquestNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # FIRE
                2: 2,  # UP
                3: 3,  # RIGHT
                4: 4,  # LEFT
                5: 5,  # DOWN
            }
        elif env_id == 'SkiingNoFrameskip-v4':
            return self._identity_action_remap(env)
        elif env_id == 'SolarisNoFrameskip-v4':
            return self._identity_action_remap(env)
        elif env_id == 'SpaceInvadersNoFrameskip-v4':
            return {
                0: 0,  # NOOP
                1: 1,  # FIRE
                2: 2,  # RIGHT
                3: 3,  # LEFT
            }
        elif env_id == 'VentureNoFrameskip-v4':
            return self._identity_action_remap(env)
        else:
            raise ValueError(
                'Has not specified action mapping for environment ID "{}"'.format(env_id))

    def _identity_action_remap(self, env):
        action_remap = {}

        for i in range(env.action_space.n):
            action_remap[i] = i

        return action_remap
