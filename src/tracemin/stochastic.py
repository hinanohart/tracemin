"""The ``[stochastic]`` extra: statistical certification under a stochastic policy.

The single-shot core never certifies. This module adds the only path to
``certified=true``: re-run the minimal set and each leave-one-out *k* times, and
mark an atom *necessary* only when removing it drops the reproduction rate with
**interval separation** (disjoint Bayesian credible intervals). Certification
requires every removable witness proven *and* the minimal set reproducing strongly.

The estimators (pass^k, Jeffreys Beta posterior, two-proportion test, interval
disjointness) mirror ``passwedge`` and are re-derived natively on numpy/scipy so
the extra installs without a git dependency; when ``passwedge`` is importable it is
preferred automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb

from tracemin.atoms import Atom, Trajectory
from tracemin.engine import MinimizeResult, Witness
from tracemin.oracle import Oracle
from tracemin.replay import OracleFn, ReplayFn, SignatureFn, Verdict, run_trial
from tracemin.signature import failure_signature

try:  # prefer the sibling library when present
    import passwedge as _pw
except Exception:  # noqa: BLE001
    _pw = None

backend = "passwedge" if _pw is not None else "native"

_PRIORS = {"jeffreys": (0.5, 0.5), "uniform": (1.0, 1.0), "haldane": (0.0, 0.0)}


@dataclass(frozen=True)
class BetaPosterior:
    """Native Beta posterior (used when passwedge is absent)."""

    a: float
    b: float

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        from scipy.stats import beta

        lo = (1.0 - level) / 2.0
        return (float(beta.ppf(lo, self.a, self.b)), float(beta.ppf(1.0 - lo, self.a, self.b)))

    def expected_pow_k(self, k: int) -> float:
        p = 1.0
        for i in range(k):
            p *= (self.a + i) / (self.a + self.b + i)
        return p


def _native_pass_pow_k(n: int, c: int, k: int, estimator: str = "unbiased") -> float:
    if estimator not in ("unbiased", "plugin"):
        raise ValueError(f"unknown estimator {estimator!r}; use 'unbiased' or 'plugin'")
    if estimator == "plugin":
        return (c / n) ** k if n else 0.0
    if k > n:
        raise ValueError("k must be <= n for the unbiased estimator")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def pass_pow_k(n: int, c: int, k: int, estimator: str = "unbiased") -> float:
    """Probability that k draws all reproduce (unbiased hypergeometric or plugin)."""
    if _pw is not None:
        return float(_pw.pass_pow_k(n, c, k, estimator=estimator))
    return _native_pass_pow_k(n, c, k, estimator)


def beta_posterior(c: int, n: int, prior: str | tuple[float, float] = "jeffreys") -> object:
    """Beta posterior from c successes in n trials (Jeffreys default). passwedge if present."""
    if _pw is not None:
        return _pw.beta_posterior(c, n, prior=prior)
    a0, b0 = _PRIORS.get(prior, (0.5, 0.5)) if isinstance(prior, str) else prior
    return BetaPosterior(a0 + c, b0 + (n - c))


def intervals_disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True iff the two closed intervals do not overlap."""
    if _pw is not None:
        return bool(_pw.intervals_disjoint(a, b))
    return a[1] < b[0] or b[1] < a[0]


def _credible(c: int, n: int, level: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    post = beta_posterior(c, n)
    ci = post.credible_interval(level)  # type: ignore[attr-defined]
    return (float(ci[0]), float(ci[1]))


def _as_oracle_fn(oracle: Oracle | OracleFn) -> OracleFn:
    classify = getattr(oracle, "classify", None)
    return classify if callable(classify) else oracle  # type: ignore[return-value]


def certify(
    result: MinimizeResult,
    trajectory: Trajectory,
    replay_fn: ReplayFn,
    oracle: Oracle | OracleFn,
    *,
    k: int = 30,
    signature_fn: SignatureFn | None = failure_signature,
    level: float = 0.95,
    tau: float = 0.5,
    error_retries: int = 2,
) -> MinimizeResult:
    """Stochastically certify a single-shot minimal result. Returns a new result.

    ``certified=True`` iff every removable witness is *proven* (removing it drops the
    reproduction rate with disjoint credible intervals) and the minimal set itself
    reproduces strongly (lower credible bound > ``tau``).
    """
    classify = _as_oracle_fn(oracle)
    reference = result.reference_signature
    by_id = trajectory.by_id()
    order_of = {a.id: a.order for a in trajectory.atoms}
    removable = set(trajectory.removable_ids)
    minimal_set = frozenset(result.minimal_ids)

    def atoms_for(ids: frozenset[str]) -> list[Atom]:
        return [by_id[i] for i in sorted(ids, key=lambda i: order_of[i])]

    def rate(ids: frozenset[str]) -> tuple[int, int]:
        c = n = 0
        for _ in range(k):
            res = None
            for _ in range(error_retries + 1):
                res = run_trial(replay_fn, classify, signature_fn, atoms_for(ids))
                if res.verdict is not Verdict.ERROR:
                    break
            if res is None or res.verdict is Verdict.ERROR:
                continue  # ERROR excluded from the denominator
            n += 1
            if res.verdict is Verdict.FAIL and (reference is None or res.signature == reference):
                c += 1
        return c, n

    c_min, n_min = rate(minimal_set)
    ci_min = _credible(c_min, n_min, level)

    witnesses: list[Witness] = []
    removable_witnesses = 0
    all_removable_proven = True
    for w in result.witnesses:
        if w.necessity == "pinned" or w.atom_id not in removable:
            witnesses.append(Witness(w.atom_id, "pinned"))
            continue
        removable_witnesses += 1
        survivors = trajectory.survivors_after_removing(removable - (minimal_set - {w.atom_id}))
        if survivors is None:
            witnesses.append(Witness(w.atom_id, "unproven"))
            all_removable_proven = False
            continue
        c_wo, n_wo = rate(frozenset(survivors))
        ci_wo = _credible(c_wo, n_wo, level)
        proven = (
            n_min > 0
            and n_wo > 0
            and (c_min / n_min) > (c_wo / n_wo)
            and intervals_disjoint(ci_min, ci_wo)
        )
        witnesses.append(Witness(w.atom_id, "proven" if proven else "unproven"))
        all_removable_proven = all_removable_proven and proven

    strong = n_min > 0 and ci_min[0] > tau
    certified = bool(removable_witnesses > 0 and all_removable_proven and strong)
    if certified:
        reason: str | None = None
    elif not strong:
        reason = "weak-reproduction"
    else:
        reason = "necessity-unproven"

    return replace(
        result,
        certified=certified,
        certified_reason=reason,
        replay_mode="stochastic",
        witnesses=tuple(witnesses),
        stats={**result.stats, "stochastic_k": k, "c_min": c_min, "n_min": n_min},
    )
