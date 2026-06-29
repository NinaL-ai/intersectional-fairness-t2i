# experiments/exp1/config.py

PROFESSIONS = [
    "doctor", "nurse", "surgeon", "psychologist",  # Healthcare
    "singer", "artist", "writer",  # Creative
    "police officer", "judge",  # Law / authority
    "scientist", "software engineer",  # Tech / science
    "teacher",  # Education
    "CEO", "journalist", "politician", "office worker",  # Business / white collar
    "chef", "waiter", "flight attendant", "receptionist", # Service / hospitality  #
    "mechanic", "cashier", "baker",  # Manual labor
    "professional athlete", "basketball player", "fitness instructor"  # Sports
]

GROUPS = [
    "neutral",
    "female", "male",
    "young", "middle aged", "elderly",
    "white", "asian", "indian", "black"
]

MODELS = [
    "Stablexl1.0",
    "Stable3.5-medium",
    "Flux-dev"
]

NUM_IMAGES_NEUTRAL = 152
NUM_IMAGES_GROUPS = 50
BATCH_SIZE_NEUTRAL = 4
BATCH_SIZE_GROUPS = 5
