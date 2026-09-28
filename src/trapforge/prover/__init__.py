"""The uniqueness prover: does an observed sample pin down exactly one hidden world?

A task family describes what its visible sample says about the hidden parameters as a
:class:`~trapforge.prover.model.ConstraintSystem` (bounded integer unknowns, linear equations,
linear congruences and a finite case split over discrete choices).
"""

from trapforge.prover.model import (
    Choice,
    ChoiceTerm,
    Congruent,
    Constraint,
    ConstraintSystem,
    Equation,
    Instance,
    InstanceRow,
    ModelError,
    Scalar,
    Unknown,
    evaluate,
)

__all__ = [
    "Choice",
    "ChoiceTerm",
    "Congruent",
    "Constraint",
    "ConstraintSystem",
    "Equation",
    "Instance",
    "InstanceRow",
    "ModelError",
    "Scalar",
    "Unknown",
    "evaluate",
]
