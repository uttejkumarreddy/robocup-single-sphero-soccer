FIELD_LENGTH = 48
FIELD_WIDTH = 33

RANDOMIZE_INITIAL_POSITIONS_PLAYERS = False
RANDOMIZE_INITIAL_POSITIONS_BALL = False

RADIUS_PLAYER = 0.365
RADIUS_BALL = 0.215

# Sphero Bolt Specs
BOLT_MIN_SPEED = 0
BOLT_MAX_SPEED = 255
BOLT_MIN_ROTATION = 0
BOLT_MAX_ROTATION = 359

# Sphero Bolt Real Specs
BOLT_REAL_MIN_SPEED = 0
BOLT_REAL_MAX_SPEED = 20 # 10 * m/s

# Soccer Ball Specs
BALL_MIN_SPEED = 0
BALL_MAX_SPEED = 20000 # from the momentum conservation equation
BALL_MIN_ROTATION = 0
BALL_MAX_ROTATION = 359

TEAM_HOME = "HOME_TEAM"  # blue players
TEAM_AWAY = "AWAY_TEAM"  # red players
NAME_BALL = "ball"

ENV_1A_0D_0K = "Environment_1A_0D_0K"

# Goal Positions
GOAL_HOME = [-FIELD_LENGTH / 2, -FIELD_WIDTH / 2]
GOAL_AWAY = [FIELD_LENGTH / 2, FIELD_WIDTH / 2]

# Checkpoint Directories
DDPG_CHKPT_DIR = "AI/DDPG/saved_models/"