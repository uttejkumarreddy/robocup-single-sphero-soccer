# Observation Space Configuration
FIELD_LENGTH = 3
FIELD_WIDTH = 3

RANDOMIZE_INITIAL_POSITIONS_PLAYERS = True
RANDOMIZE_INITIAL_POSITIONS_BALL = True

RADIUS_PLAYER = 0.365
RADIUS_BALL = 0.215

# Sphero Bolt Specs - API
# Useful for scaling ENV values to API values
BOLT_MIN_SPEED = 0
BOLT_MAX_SPEED = 255
BOLT_MIN_ROTATION = 0
BOLT_MAX_ROTATION = 359

# Sphero Bolt Specs - MuJoCo
BOLT_ENV_MIN_SPEED = 10	# Min 0 unit/sec
BOLT_ENV_MAX_SPEED = 20 # Max 20 unit/sec
BOLT_ENV_MIN_ROTATION = 0
BOLT_ENV_MAX_ROTATION = 359
BOLT_ENV_MASS = 200

# Soccer Ball Specs -MuJoCo
BALL_ENV_MIN_SPEED = 0
BALL_ENV_MAX_SPEED = 1500 # derived from the momentum conservation equation
BALL_ENV_MIN_ROTATION = 0
BALL_ENV_MAX_ROTATION = 359
BALL_ENV_MASS = 2.77

TEAM_HOME = "HOME_TEAM"  # blue players
TEAM_AWAY = "AWAY_TEAM"  # red players
NAME_BALL = "ball"

ENV_1A_0D_0K = "Environment_1A_0D_0K"

# Goal Positions
GOAL_HOME = [-FIELD_LENGTH / 2, -FIELD_WIDTH / 2]
GOAL_AWAY = [FIELD_LENGTH / 2, FIELD_WIDTH / 2]

# Checkpoint Directories
DDPG_CHKPT_DIR = "AI/DDPG/saved_models/"