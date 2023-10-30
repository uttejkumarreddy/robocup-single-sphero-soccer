# Observation Space
OBSERVATION_SPACE = {
    'XS': {
        'FIELD_DIMENSIONS': [6, 4.125],
        'GOAL_HOME': [-3, -2.0625],
        'GOAL_AWAY': [3, 2.0625],
    },
    'S': {
        'FIELD_DIMENSIONS': [12, 8.25],
        'GOAL_HOME': [-6, -4.125],
        'GOAL_AWAY': [6, 4.125],
    },
    'M': {
        'FIELD_DIMENSIONS': [24, 16.5],
        'GOAL_HOME': [-12, -8.25],
        'GOAL_AWAY': [12, 8.25],
    },
    'L': {
        'FIELD_DIMENSIONS': [48, 33],
        'GOAL_HOME': [-24, -16.5],
        'GOAL_AWAY': [24, 16.5],
    },
}

RANDOMIZE_INITIAL_POSITIONS_PLAYERS = True
RANDOMIZE_INITIAL_POSITIONS_BALL = True

RADIUS_PLAYER = 0.365
RADIUS_BALL = 0.215

# Sphero Bolt and Soccer Ball Specs - API
# Useful for scaling ENV values to API values
BOLT_MIN_SPEED = 0
BOLT_MAX_SPEED = 255
BOLT_MIN_ROTATION = 0
BOLT_MAX_ROTATION = 359
BOLT_MASS = 200

BALL_MIN_SPEED = 0
BALL_MAX_SPEED = 20000 # derived from the momentum conservation equation
BALL_MIN_ROTATION = 0
BALL_MAX_ROTATION = 359
BALL_MASS = 2.77

# Sphero Bolt Specs - MuJoCo
ENV_BOLT_MIN_SPEED = 0	# Min 0 unit/sec
ENV_BOLT_MAX_SPEED = 20 # Max 20 unit/sec
ENV_BOLT_MIN_ROTATION = 0
ENV_BOLT_MAX_ROTATION = 359
ENV_BOLT_MASS = 200

# Soccer Ball Specs - MuJoCo
ENV_BALL_MIN_SPEED = 0
ENV_BALL_MAX_SPEED = 1500 # derived from the momentum conservation equation
ENV_BALL_MIN_ROTATION = 0
ENV_BALL_MAX_ROTATION = 359
ENV_BALL_MASS = 2.77

TEAM_HOME = "HOME_TEAM"  # blue players
TEAM_AWAY = "AWAY_TEAM"  # red players
NAME_BALL = "ball"

ENV_1A_0D_0K = "Environment_1A_0D_0K"

