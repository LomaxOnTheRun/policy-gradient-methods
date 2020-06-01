ENVIRONMENTS_INFO = {
    # Classic control
    # https://github.com/openai/gym/wiki/Leaderboard#classic-control
    "CartPole-v0": {
        "solvable": True,
        "name": "CartPole-v0",
        "success_score": 195,
        "success_episodes": 100,
    },
    "CartPole-v1": {
        "solvable": True,
        "name": "CartPole-v1",
        "success_score": 495,
        "success_episodes": 100,
    },
    "MountainCar-v0": {
        "solvable": True,
        "name": "MountainCar-v0",
        "success_score": -110,
        "success_episodes": 100,
    },
    "MountainCarContinuous-v0": {
        "solvable": True,
        "name": "MountainCarContinuous-v0",
        "success_score": 90,
        "success_episodes": 100,
    },
    "Pendulum-v0": {
        "solvable": False,
        "name": "Pendulum-v0",
        "success_score": -150,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    "Acrobot-v1": {
        "solvable": False,
        "name": "Acrobot-v1",
        "success_score": -100,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    # Box2D
    # https://github.com/openai/gym/wiki/Leaderboard#box2d
    "LunarLander-v2": {
        "solvable": True,
        "name": "LunarLander-v2",
        "success_score": 200,
        "success_episodes": 100,
    },
    "LunarLanderContinuous-v2": {
        "solvable": True,
        "name": "LunarLanderContinuous-v2",
        "success_score": 200,
        "success_episodes": 100,
    },
    "BipedalWalker-v3": {
        "solvable": True,
        "name": "BipedalWalker-v3",
        "success_score": 300,
        "success_episodes": 100,
    },
    "BipedalWalkerHardcore-v3": {
        "solvable": True,
        "name": "BipedalWalkerHardcore-v3",
        "success_score": 300,
        "success_episodes": 100,
    },
    "CarRacing-v0": {
        "solvable": True,
        "name": "CarRacing-v0",
        "success_score": 900,
        "success_episodes": 100,
    },
    # MuJoCo
    # https://github.com/openai/gym/wiki/Leaderboard#mujoco
    "Walker2d-v1": {
        "solvable": False,
        "name": "Walker2d-v1",
        "success_score": 1000,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    "Walker2d-v2": {
        "solvable": False,
        "name": "Walker2d-v2",
        "success_score": 1000,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    "Ant-v1": {
        "solvable": True,
        "name": "Ant-v1",
        "success_score": 6000,
        "success_episodes": 100,
    },
    # PyGame Learning Environment
    # https://github.com/openai/gym/wiki/Leaderboard#pygame-learning-environment
    "FlappyBird-v0": {
        "solvable": False,
        "name": "FlappyBird-v0",
        "success_score": 10,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    # Atari games
    # https://github.com/openai/gym/wiki/Leaderboard#atari-games
    # All scores are human scores taken from https://arxiv.org/pdf/2003.13350.pdf
    "Breakout-v0": {
        "solvable": False,
        "name": "Breakout-v0",
        "success_score": 30.5,
        "success_episodes": 100,
    },
    "Pong-v0": {
        "solvable": False,
        "name": "Pong-v0",
        "success_score": 14.6,
        "success_episodes": 100,
    },
    "MsPacman-v0": {
        "solvable": False,
        "name": "MsPacman-v0",
        "success_score": 6951.60,
        "success_episodes": 100,
    },
    "SpaceInvaders-v0": {
        "solvable": False,
        "name": "SpaceInvaders-v0",
        "success_score": 1668.7,
        "success_episodes": 100,
    },
    "Seaquest-v0": {
        "solvable": False,
        "name": "Seaquest-v0",
        "success_score": 42054.7,
        "success_episodes": 100,
    },
    # Snake
    # https://github.com/openai/gym/wiki/Leaderboard#snake-v0
    "Snake-v0": {
        "solvable": False,
        "name": "Snake-v0",
        "success_score": 0.4,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    # Doom
    # https://github.com/openai/gym/wiki/Leaderboard#doom
    "Doom Basic Scenario": {
        "solvable": True,
        "name": "Doom Basic Scenario",
        "success_score": 70,
        "success_episodes": 100,
    },
    "Doom Deadly Corridor Scenario": {
        "solvable": False,
        "name": "Doom Deadly Corridor Scenario",
        "success_score": 500,  # Based on leaderboard scores
        "success_episodes": 100,
    },
    # Toy text
    # https://github.com/openai/gym/wiki/Leaderboard#toy-text
    # These have been omitted as the solved conditions are not simply to achieve a
    # running average of a certain score.
}


def get_env_info(env_name):
    """
    Return meta-data about env (inc. success steps and score).
    """
    if env_name not in ENVIRONMENTS_INFO:
        raise Exception(f"Environment {env_name} unknown")

    return ENVIRONMENTS_INFO[env_name]
