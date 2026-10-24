"""Render a computation graph as Mermaid (renders on GitHub) or Graphviz DOT."""

from __future__ import annotations

from .tensor import Tensor


def _walk(root: Tensor):
    nodes: list[Tensor] = []
    edges: list[tuple[Tensor, Tensor]] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        nodes.append(t)
        if t._ctx is not None:
            for parent in t._ctx.parents:
                edges.append((parent, t))
                stack.append(parent)
    return nodes[::-1], edges[::-1]


def _label(t: Tensor, names: dict[int, str]) -> str:
    shape = "×".join(map(str, t.shape)) or "scalar"
    if t._ctx is not None:
        return f"{t._ctx.name}<br/>{shape}"
    return f"{names.get(id(t), 'param' if t.requires_grad else 'const')}<br/>{shape}"


def to_mermaid(root: Tensor, params: dict | None = None, direction: str = "LR") -> str:
    """Mermaid flowchart of every op that produced `root`.

    Pass `params=dict(model.named_parameters())` to label weights by name.
    """
    names = {id(p): name for name, p in (params or {}).items()}
    nodes, edges = _walk(root)
    ids = {id(t): f"n{i}" for i, t in enumerate(nodes)}
    lines = [f"flowchart {direction}"]
    for t in nodes:
        label = _label(t, names)
        if t._ctx is not None:
            lines.append(f'  {ids[id(t)]}(["{label}"])')
        elif t.requires_grad:
            lines.append(f'  {ids[id(t)]}["{label}"]')
        else:
            lines.append(f'  {ids[id(t)]}[/"{label}"/]')
    for src, dst in edges:
        lines.append(f"  {ids[id(src)]} --> {ids[id(dst)]}")
    lines.append(f"  style {ids[id(root)]} stroke-width:3px")
    return "\n".join(lines)


def to_dot(root: Tensor, params: dict | None = None) -> str:
    names = {id(p): name for name, p in (params or {}).items()}
    nodes, edges = _walk(root)
    ids = {id(t): f"n{i}" for i, t in enumerate(nodes)}
    lines = ["digraph G {", "  rankdir=LR;", "  node [fontname=Helvetica, fontsize=10];"]
    for t in nodes:
        label = _label(t, names).replace("<br/>", "\\n")
        shape = "ellipse" if t._ctx is not None else ("box" if t.requires_grad else "note")
        lines.append(f'  {ids[id(t)]} [label="{label}", shape={shape}];')
    for src, dst in edges:
        lines.append(f"  {ids[id(src)]} -> {ids[id(dst)]};")
    lines.append("}")
    return "\n".join(lines)
