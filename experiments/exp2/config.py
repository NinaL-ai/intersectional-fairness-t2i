# experiments/exp1/config.py

PROFESSIONS = [
    "doctor", # Healthcare
    "writer",  # Creative
    "judge",  # Law / authority
    "software engineer",  # Tech / science
    "teacher",  # Education
    "CEO",  # Business / white collar
    "chef",  # Service / hospitality  #
    "cashier",  # Manual labor
    "professional athlete"  # Sports
]

GROUPS = [
    "neutral",
    "female", "male",
    "young", "middle aged", "elderly",
    "20 years old", "40 years old", "75 years old",
    "white", "asian", "indian", "black"
]

MODELS = [
    "Stablexl1.0",
    "Stable3.5-medium",
    "Flux-dev"
]

NUM_IMAGES = 50
BATCH_SIZE = 5
