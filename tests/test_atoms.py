from __future__ import annotations

from tracemin.atoms import Atom, AtomKind, Trajectory, content_id


def _atom(payload, *, produces=(), requires=(), pinned=False, order=0):
    return Atom.make(
        AtomKind.MESSAGE, payload, produces=produces, requires=requires, pinned=pinned, order=order
    )


def test_content_id_is_deterministic_and_content_addressed():
    a = content_id(AtomKind.TOOL_DEF, {"name": "search"}, ["search"], [])
    b = content_id(AtomKind.TOOL_DEF, {"name": "search"}, ["search"], [])
    c = content_id(AtomKind.TOOL_DEF, {"name": "other"}, ["other"], [])
    assert a == b
    assert a != c
    assert a.startswith("too-")


def test_removable_excludes_pinned_in_order():
    t = Trajectory.of(
        [
            _atom("a", order=0, pinned=True),
            _atom("b", order=1),
            _atom("c", order=2),
        ]
    )
    assert set(t.pinned_ids) == {t.atoms[0].id}
    assert t.removable_ids == (t.atoms[1].id, t.atoms[2].id)


def test_well_formed_requires_internal_producer_present():
    tool = _atom({"tool": "search"}, produces=["search"], order=0)
    caller = _atom({"call": "search"}, requires=["search"], order=1)
    t = Trajectory.of([tool, caller])
    # Full set is well-formed.
    assert t.is_well_formed({tool.id, caller.id})
    # Dropping the producer makes the caller dangling.
    assert not t.is_well_formed({caller.id})
    # Ambient (non-internal) requires never block.
    ambient = _atom({"call": "weather"}, requires=["weather"], order=2)
    t2 = Trajectory.of([ambient])
    assert t2.is_well_formed({ambient.id})


def test_closure_remove_cascades_to_dependents():
    tool = _atom({"tool": "search"}, produces=["search"], order=0)
    caller = _atom({"call": "search"}, requires=["search"], order=1)
    other = _atom("unrelated", order=2)
    t = Trajectory.of([tool, caller, other])
    closed = t.closure_remove({tool.id})
    assert closed is not None
    # removing the tool must also remove its dependent caller
    assert {tool.id, caller.id} <= closed
    assert other.id not in closed


def test_closure_remove_blocked_by_pinned_dependent():
    tool = _atom({"tool": "search"}, produces=["search"], order=0)
    pinned_caller = _atom({"call": "search"}, requires=["search"], pinned=True, order=1)
    t = Trajectory.of([tool, pinned_caller])
    # Cannot remove the tool because that would force removing a pinned dependent.
    assert t.closure_remove({tool.id}) is None
    assert t.survivors_after_removing({tool.id}) is None


def test_survivors_keep_pinned_and_are_well_formed():
    tool = _atom({"tool": "search"}, produces=["search"], order=0)
    caller = _atom({"call": "search"}, requires=["search"], order=1)
    keep = _atom("keep", pinned=True, order=2)
    t = Trajectory.of([tool, caller, keep])
    survivors = t.survivors_after_removing({caller.id})
    assert survivors is not None
    assert keep.id in survivors and tool.id in survivors
    assert t.is_well_formed(survivors)
