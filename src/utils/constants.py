"""League codes, threshold defaults, and cross-source mappings."""

# Default Over/Under thresholds
GOAL_THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]
CORNER_THRESHOLDS = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

# League code mappings across data sources
LEAGUE_CODES = {
    "EPL": {
        "name": "Premier League",
        "country": "England",
        "football_data_id": "PL",
        "api_football_id": 39,
        "fbref_slug": "Premier-League",
    },
    "LA_LIGA": {
        "name": "La Liga",
        "country": "Spain",
        "football_data_id": "PD",
        "api_football_id": 140,
        "fbref_slug": "La-Liga",
    },
    "BUNDESLIGA": {
        "name": "Bundesliga",
        "country": "Germany",
        "football_data_id": "BL1",
        "api_football_id": 78,
        "fbref_slug": "Bundesliga",
    },
    "SERIE_A": {
        "name": "Serie A",
        "country": "Italy",
        "football_data_id": "SA",
        "api_football_id": 135,
        "fbref_slug": "Serie-A",
    },
    "LIGUE_1": {
        "name": "Ligue 1",
        "country": "France",
        "football_data_id": "FL1",
        "api_football_id": 61,
        "fbref_slug": "Ligue-1",
    },
}

# International competition IDs (API-Football)
INTERNATIONAL_COMPS = {
    "WORLD_CUP": {"name": "FIFA World Cup", "api_football_id": 1},
    "EUROS": {"name": "UEFA European Championship", "api_football_id": 4},
    "NATIONS_LEAGUE": {"name": "UEFA Nations League", "api_football_id": 5},
    "FRIENDLIES": {"name": "International Friendlies", "api_football_id": 10},
}

# Feature engineering defaults
DEFAULT_ROLLING_WINDOW = 10
MIN_MATCHES_REQUIRED = 5

# Model defaults
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 4
DEFAULT_LEARNING_RATE = 0.1

# Maximum goals/corners for probability distribution calculations
MAX_GOALS = 10
MAX_CORNERS = 25
