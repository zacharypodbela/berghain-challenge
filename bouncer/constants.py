"""
Constants for Berghain Challenge scenarios
These are extracted from actual API responses and are consistent across all games
"""

from datetime import timedelta
from typing import Any

CAPACITY = 1000  # Max people that can be admitted
REJECTION_LIMIT = 20000  # Max rejections before game fails

RATE_LIMIT_TIME = timedelta(
    minutes=15, seconds=30
)  # 15 minute rolling window + 30s buffer
RATE_LIMIT_N = 10  # Max 10 RemoteGames per window

SCENARIO_CONFIGS: dict[int, dict[str, Any]] = {
    1: {
        "constraints": [
            {"attribute": "young", "minCount": 600},
            {"attribute": "well_dressed", "minCount": 600},
        ],
        "attribute_statistics": {
            "relativeFrequencies": {"well_dressed": 0.3225, "young": 0.3225},
            "correlations": {
                "well_dressed": {"well_dressed": 1, "young": 0.18304299322062992},
                "young": {"well_dressed": 0.18304299322062992, "young": 1},
            },
        },
    },
    2: {
        "constraints": [
            {"attribute": "techno_lover", "minCount": 650},
            {"attribute": "well_connected", "minCount": 450},
            {"attribute": "creative", "minCount": 300},
            {"attribute": "berlin_local", "minCount": 750},
        ],
        "attribute_statistics": {
            "relativeFrequencies": {
                "techno_lover": 0.6265000000000001,
                "well_connected": 0.4700000000000001,
                "creative": 0.06227,
                "berlin_local": 0.398,
            },
            "correlations": {
                "techno_lover": {
                    "techno_lover": 1,
                    "well_connected": -0.4696169332674324,
                    "creative": 0.09463317039891586,
                    "berlin_local": -0.6549403815606182,
                },
                "well_connected": {
                    "techno_lover": -0.4696169332674324,
                    "well_connected": 1,
                    "creative": 0.14197259140471485,
                    "berlin_local": 0.5724067808436452,
                },
                "creative": {
                    "techno_lover": 0.09463317039891586,
                    "well_connected": 0.14197259140471485,
                    "creative": 1,
                    "berlin_local": 0.14446459505650772,
                },
                "berlin_local": {
                    "techno_lover": -0.6549403815606182,
                    "well_connected": 0.5724067808436452,
                    "creative": 0.14446459505650772,
                    "berlin_local": 1,
                },
            },
        },
    },
    3: {
        "constraints": [
            {"attribute": "underground_veteran", "minCount": 500},
            {"attribute": "international", "minCount": 650},
            {"attribute": "fashion_forward", "minCount": 550},
            {"attribute": "queer_friendly", "minCount": 250},
            {"attribute": "vinyl_collector", "minCount": 200},
            {"attribute": "german_speaker", "minCount": 800},
        ],
        "attribute_statistics": {
            "relativeFrequencies": {
                "underground_veteran": 0.6794999999999999,
                "international": 0.5735,
                "fashion_forward": 0.6910000000000002,
                "queer_friendly": 0.04614,
                "vinyl_collector": 0.044539999999999996,
                "german_speaker": 0.4565000000000001,
            },
            "correlations": {
                "underground_veteran": {
                    "underground_veteran": 1,
                    "international": -0.08110175777152992,
                    "fashion_forward": -0.1696563475505309,
                    "queer_friendly": 0.03719928376753885,
                    "vinyl_collector": 0.07223521156389842,
                    "german_speaker": 0.11188766703422799,
                },
                "international": {
                    "underground_veteran": -0.08110175777152992,
                    "international": 1,
                    "fashion_forward": 0.375711059360155,
                    "queer_friendly": 0.0036693314388711686,
                    "vinyl_collector": -0.03083247098181075,
                    "german_speaker": -0.7172529382519395,
                },
                "fashion_forward": {
                    "underground_veteran": -0.1696563475505309,
                    "international": 0.375711059360155,
                    "fashion_forward": 1,
                    "queer_friendly": -0.0034530926793377476,
                    "vinyl_collector": -0.11024719606358546,
                    "german_speaker": -0.3521024461597403,
                },
                "queer_friendly": {
                    "underground_veteran": 0.03719928376753885,
                    "international": 0.0036693314388711686,
                    "fashion_forward": -0.0034530926793377476,
                    "queer_friendly": 1,
                    "vinyl_collector": 0.47990640803167306,
                    "german_speaker": 0.04797381132680503,
                },
                "vinyl_collector": {
                    "underground_veteran": 0.07223521156389842,
                    "international": -0.03083247098181075,
                    "fashion_forward": -0.11024719606358546,
                    "queer_friendly": 0.47990640803167306,
                    "vinyl_collector": 1,
                    "german_speaker": 0.09984452286269897,
                },
                "german_speaker": {
                    "underground_veteran": 0.11188766703422799,
                    "international": -0.7172529382519395,
                    "fashion_forward": -0.3521024461597403,
                    "queer_friendly": 0.04797381132680503,
                    "vinyl_collector": 0.09984452286269897,
                    "german_speaker": 1,
                },
            },
        },
    },
}
