"""The uniqueness prover: does an observed sample pin down exactly one hidden world?

A task family describes what its visible sample says about the hidden parameters as a
:class:`~trapforge.prover.model.ConstraintSystem` (bounded integer unknowns, linear equations,
linear congruences and a finite case split over discrete choices). :func:`prove` decides
exactly how many hidden worlds fit and returns a :class:`UniqueProof`, an :class:`Ambiguity`
(with a counterexample pair) or an :class:`Infeasible`, each with a JSON certificate that
:func:`check_certificate` re-checks independently. :func:`gate` extends a generated sample
with further observations until the proof is unique.
"""

from trapforge.prover.certificate import CertificateCheck, check_certificate
from trapforge.prover.gating import GateError, GateResult, gate
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
from trapforge.prover.solver import (
    DEFAULT_CAP,
    Ambiguity,
    Infeasible,
    Proof,
    UniqueProof,
    certificate_json,
    prove,
    reduce_case,
)

__all__ = [
    "DEFAULT_CAP",
    "Ambiguity",
    "CertificateCheck",
    "Choice",
    "ChoiceTerm",
    "Congruent",
    "Constraint",
    "ConstraintSystem",
    "Equation",
    "GateError",
    "GateResult",
    "Infeasible",
    "Instance",
    "InstanceRow",
    "ModelError",
    "Proof",
    "Scalar",
    "UniqueProof",
    "Unknown",
    "certificate_json",
    "check_certificate",
    "evaluate",
    "gate",
    "prove",
    "reduce_case",
]
