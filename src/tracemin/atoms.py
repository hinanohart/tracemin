"""Typed trajectory model: content-addressed atoms and a dependency DAG.

A failed agent run is normalized into a flat sequence of heterogeneous *atoms*
(messages, tool definitions, retrieved files, instructions). Each atom is a
candidate for removal during reduction. Atoms declare the symbols they
``produces`` and the symbols they ``requires``; together these induce a
dependency DAG used to keep every reduction candidate well-formed.

Identity is content-addressed: two atoms with the same kind, payload and
dependency edges share an id (and are interchangeable for reduction).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


class AtomKind(str, Enum):
    """The four context categories tracemin reduces over."""

    MESSAGE = "message"
    TOOL_DEF = "tool_def"
    RETRIEVED_FILE = "retrieved_file"
    INSTRUCTION = "instruction"


def _canonical(
    kind: AtomKind, payload: object, produces: Iterable[str], requires: Iterable[str]
) -> str:
    """Stable canonical serialization used for content addressing."""
    blob = json.dumps(
        {
            "kind": kind.value,
            "payload": payload,
            "produces": sorted(set(produces)),
            "requires": sorted(set(requires)),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return blob


def content_id(
    kind: AtomKind, payload: object, produces: Iterable[str], requires: Iterable[str]
) -> str:
    """Content-addressed atom id (12-hex prefix of sha256 over canonical content)."""
    digest = hashlib.sha256(
        _canonical(kind, payload, produces, requires).encode("utf-8")
    ).hexdigest()
    return f"{kind.value[:3]}-{digest[:12]}"


@dataclass(frozen=True)
class Atom:
    """A single removable unit of agent context.

    ``produces`` / ``requires`` are sets of opaque symbol names (e.g. a tool name,
    a file path, a variable). They define the dependency DAG: an atom that
    ``requires`` a symbol can only appear in a reduction together with some atom
    that ``produces`` it.
    """

    id: str
    kind: AtomKind
    payload: object
    produces: frozenset[str] = field(default_factory=frozenset)
    requires: frozenset[str] = field(default_factory=frozenset)
    pinned: bool = False
    order: int = -1  # -1 = position not yet assigned; Trajectory.of fills it by sequence index

    @staticmethod
    def make(
        kind: AtomKind,
        payload: object,
        *,
        produces: Iterable[str] = (),
        requires: Iterable[str] = (),
        pinned: bool = False,
        order: int = -1,
        id: str | None = None,
    ) -> Atom:
        prod = frozenset(produces)
        req = frozenset(requires)
        return Atom(
            id=id or content_id(kind, payload, prod, req),
            kind=kind,
            payload=payload,
            produces=prod,
            requires=req,
            pinned=pinned,
            order=order,
        )


@dataclass(frozen=True)
class Trajectory:
    """An ordered collection of atoms with dependency-DAG queries.

    The DAG treats a required symbol as a constraint only if it is *internal*
    (produced by some atom in the trajectory). Required symbols produced by no
    atom are ambient givens and never block a reduction.
    """

    atoms: tuple[Atom, ...]

    def __post_init__(self) -> None:
        if len({a.id for a in self.atoms}) != len(self.atoms):
            # Duplicate content ids are allowed conceptually, but our id->atom maps
            # require uniqueness; dedupe deterministically by first occurrence.
            seen: dict[str, Atom] = {}
            for a in self.atoms:
                seen.setdefault(a.id, a)
            object.__setattr__(self, "atoms", tuple(seen.values()))

    @staticmethod
    def of(atoms: Sequence[Atom]) -> Trajectory:
        ordered = tuple(
            a
            if a.order >= 0  # keep an explicitly assigned order (including 0), else use position
            else Atom.make(
                a.kind,
                a.payload,
                produces=a.produces,
                requires=a.requires,
                pinned=a.pinned,
                order=i,
                id=a.id,
            )
            for i, a in enumerate(atoms)
        )
        return Trajectory(ordered)

    # --- basic maps -----------------------------------------------------------

    def by_id(self) -> Mapping[str, Atom]:
        return {a.id: a for a in self.atoms}

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.atoms)

    @property
    def pinned_ids(self) -> frozenset[str]:
        return frozenset(a.id for a in self.atoms if a.pinned)

    @property
    def removable_ids(self) -> tuple[str, ...]:
        """Non-pinned atom ids in stable order — the elements ddmin operates on."""
        return tuple(a.id for a in sorted(self.atoms, key=lambda a: a.order) if not a.pinned)

    @property
    def internal_symbols(self) -> frozenset[str]:
        out: set[str] = set()
        for a in self.atoms:
            out |= a.produces
        return frozenset(out)

    # --- well-formedness & closure -------------------------------------------

    def _produced_by(self, ids: Iterable[str]) -> frozenset[str]:
        by = self.by_id()
        out: set[str] = set()
        for i in ids:
            out |= by[i].produces
        return frozenset(out)

    def is_well_formed(self, subset_ids: Iterable[str]) -> bool:
        """True iff every internal symbol required within the subset is produced within it."""
        ids = set(subset_ids)
        by = self.by_id()
        produced = self._produced_by(ids)
        internal = self.internal_symbols
        for i in ids:
            for sym in by[i].requires:
                if sym in internal and sym not in produced:
                    return False
        return True

    def closure_remove(self, remove_ids: Iterable[str]) -> frozenset[str] | None:
        """Expand a removal set to the smallest superset that keeps the remainder WF.

        Returns the full set of ids to remove, or ``None`` if keeping the remainder
        well-formed would require removing a pinned atom (caller must reject).
        """
        by = self.by_id()
        all_ids = set(by)
        internal = self.internal_symbols
        removed = set(remove_ids)
        changed = True
        while changed:
            changed = False
            remaining = all_ids - removed
            produced = self._produced_by(remaining)
            for i in list(remaining):
                atom = by[i]
                dangling = any(s in internal and s not in produced for s in atom.requires)
                if dangling:
                    if atom.pinned:
                        return None
                    removed.add(i)
                    changed = True
        return frozenset(removed)

    def survivors_after_removing(self, remove_ids: Iterable[str]) -> frozenset[str] | None:
        """Atom ids that remain (and are well-formed) after a closure-aware removal."""
        closed = self.closure_remove(remove_ids)
        if closed is None:
            return None
        survivors = frozenset(self.by_id()) - closed
        # Pinned atoms must always survive.
        if not self.pinned_ids <= survivors:
            return None
        return survivors
