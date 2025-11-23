"""Shared constants for LoKI project.

Contains datasets, integration methods, and other configuration constants
used across multiple modules.
"""

# MMLU benchmark subject datasets
MMLU_ALL_SETS = [
    "college_biology",
    "professional_law",
    "clinical_knowledge",
    "college_medicine",
    "business_ethics",
    "jurisprudence",
    "college_physics",
    "elementary_mathematics",
    "professional_medicine",
    "high_school_chemistry",
    "high_school_macroeconomics",
    "college_mathematics",
    "medical_genetics",
    "high_school_physics",
    "high_school_microeconomics",
    "professional_accounting",
    "college_computer_science",
    "high_school_world_history",
    "global_facts",
    "high_school_european_history",
    "high_school_government_and_politics",
    "virology",
    "econometrics",
    "college_chemistry",
    "conceptual_physics",
    "professional_psychology",
    "high_school_us_history",
    "abstract_algebra",
    "high_school_psychology",
    "miscellaneous",
    "marketing",
    "anatomy",
    "nutrition",
    "high_school_statistics",
    "sociology",
    "management",
    "electrical_engineering",
    "human_aging",
    "moral_scenarios",
    "us_foreign_policy",
    "high_school_computer_science",
    "high_school_geography",
    "high_school_biology",
    "moral_disputes",
    "public_relations",
    "security_studies",
    "prehistory",
    "computer_security",
    "machine_learning",
    "human_sexuality",
    "world_religions",
    "logical_fallacies",
    "philosophy",
    "astronomy",
    "formal_logic",
    "international_law",
]

# Captum LayerIntegratedGradients integration methods
INTEGRATION_METHODS = [
    "riemann_trapezoid",  # Default: trapezoidal rule approximation
    "gausslegendre",  # Gauss-Legendre quadrature
    "riemann_left",  # Left Riemann sum
    "riemann_right",  # Right Riemann sum
    "riemann_middle",  # Middle Riemann sum
]

# Default configuration values
DEFAULT_IG_STEPS = 7
DEFAULT_IG_METHOD = "riemann_trapezoid"
DEFAULT_HDF5_COMPRESSION = "gzip"
DEFAULT_HDF5_COMPRESSION_OPTS = 4
