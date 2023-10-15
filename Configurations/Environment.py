# Observation Space Configuration
FIELD_LENGTH = 3
FIELD_WIDTH = 3

RANDOMIZE_INITIAL_POSITIONS_PLAYERS = True
RANDOMIZE_INITIAL_POSITIONS_BALL = True

RADIUS_PLAYER = 0.365
RADIUS_BALL = 0.215

# Sphero Bolt Specs - API
BOLT_MIN_SPEED = 0
BOLT_MAX_SPEED = 255
BOLT_MIN_ROTATION = 0
BOLT_MAX_ROTATION = 359

# Sphero Bolt Specs - MuJoCo
BOLT_ENV_MIN_SPEED = 10	# Min 0 unit/sec
BOLT_ENV_MAX_SPEED = 20 # Max 20 unit/sec

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