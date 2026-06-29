# generation/models/loader.py

from .stablexl import SDXLGenerator
from .stable35 import Stable35Generator
from .flux_dev import FluxDevGenerator


def load_model(model_name):

    if model_name == "Stablexl1.0":
        return SDXLGenerator()

    if model_name == "Stable3.5-medium":
        return Stable35Generator()

    if model_name == "Flux-dev":
        return FluxDevGenerator()

    raise ValueError(f"Unknown model: {model_name}")