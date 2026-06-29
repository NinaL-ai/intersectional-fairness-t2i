# experiments/exp1/config.py

# PROFESSIONS = [
#     "doctor", "nurse", "surgeon", "psychologist",  # Healthcare
#     "singer", "artist", "writer",  # Creative
#     "police officer", "judge",  # Law / authority
#     "scientist", "software engineer",  # Tech / science
#     "teacher",  # Education
#     "CEO", "journalist", "politician", "office worker",  # Business / white collar
#     "chef", "waiter", "flight attendant", "receptionist", # Service / hospitality  #
#     "mechanic", "cashier", "baker",  # Manual labor
#     "professional athlete", "basketball player", "fitness instructor"  # Sports
# ]

PROFESSIONS = [
    # "doctor", # Healthcare
    "writer",  # Creative
    "judge",  # Law / authority
    "software engineer",  # Tech / science
    "teacher",  # Education
    "CEO",  # Business / white collar
    "chef",  # Service / hospitality  #
    "cashier",  # Manual labor
    "professional athlete"  # Sports
]

GROUPS = {
    "gender": ["female", "male"],
    "age": ["young", "middle aged", "elderly"],
    "race": ["white", "asian", "indian", "black"]
}

MODELS = [
    # "Stablexl1.0",
    "Stable3.5-medium",
    "Flux-dev"
]

NUM_IMAGES = 150
BATCH_SIZE = 1
