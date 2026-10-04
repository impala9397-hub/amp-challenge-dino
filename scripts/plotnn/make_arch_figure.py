"""Generator architecture figure in the PlotNeuralNet style (3D layer blocks).

Requires a local clone of PlotNeuralNet (MIT, https://github.com/HarisIqbal88/PlotNeuralNet;
only its `layers/` TikZ styles are used) and a LaTeX engine (tested with tectonic 0.17).
PlotNeuralNet is not vendored here.

    PLOTNN_DIR=/path/to/PlotNeuralNet python scripts/plotnn/make_arch_figure.py

Writes docs/figures/generator.{pdf,png}. Every number in the figure comes from
`src/dino_amp/model.py` and `src/dino_amp/tokenizer.py`; nothing here is a
separate source of truth. PNG rasterisation uses macOS `qlmanage`; on other
systems convert the PDF with any tool.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
PLOTNN = Path(os.environ.get("PLOTNN_DIR", "")).expanduser()
SCALE = 0.2  # PlotNeuralNet box scale (layers/Box.sty)

COLORS = r"""
\definecolor{embc}{RGB}{176,156,230}
\definecolor{attnc}{RGB}{130,205,170}
\definecolor{attnbandc}{RGB}{46,150,110}
\definecolor{ffnc}{RGB}{255,214,140}
\definecolor{normc}{RGB}{204,72,52}
\definecolor{headc}{RGB}{200,90,170}
\definecolor{outc}{RGB}{130,40,120}
\definecolor{inputc}{RGB}{215,215,215}
\definecolor{sumc}{RGB}{60,180,110}
"""


def head() -> str:
    layers = (PLOTNN / "layers").as_posix() + "/"
    return r"""\documentclass[border=14pt, multi, tikz]{standalone}
\usepackage{import}
\usepackage{amsmath,amssymb}
\subimport{""" + layers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d,calc}
""" + COLORS + r"""
\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
\tikzstyle{lab}=[font=\large, align=center, anchor=north]
\tikzstyle{note}=[font=\large, align=left]
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}
"""


def end() -> str:
    return "\n\\end{tikzpicture}\n\\end{document}\n"


def box(name, at, fill, h, d, w, xlabel="", zlabel="", offset="(0,0,0)", opacity=0.9):
    return rf"""
\pic[shift={{{offset}}}] at {at}
    {{Box={{name={name}, caption={{ }}, xlabel={{{{{xlabel}, }}}}, zlabel={{{zlabel}}},
           fill={fill}, opacity={opacity}, height={h}, width={w}, depth={d}}}}};"""


def banded(name, at, fill, band, h, d, w1, w2, xl1="", xl2="", zlabel="", offset="(0,0,0)", opacity=0.9):
    return rf"""
\pic[shift={{{offset}}}] at {at}
    {{RightBandedBox={{name={name}, caption={{ }}, xlabel={{{{{xl1}, {xl2}}}}}, zlabel={{{zlabel}}},
           fill={fill}, bandfill={band}, opacity={opacity}, height={h}, width={{{w1}, {w2}}}, depth={d}}}}};"""


def ball(name, at, logo, radius=2.0, offset="(0,0,0)"):
    return rf"""
\pic[shift={{{offset}}}] at {at}
    {{Ball={{name={name}, fill=sumc, opacity=0.7, radius={radius}, logo={{{logo}}}}}}};"""


def cap(name, text, depth, width_cm=4.0, dy=-1.25, dx=0.0):
    """Label under a box: anchored below the front-bottom edge (depth = unscaled box depth)."""
    return rf"""
\node[lab, text width={width_cm}cm] at ($({name}-south)+({dx},{dy},{depth * SCALE / 2:.2f})$) {{{text}}};"""


def conn(a, b):
    return rf"""
\draw [connection] ({a}-east) -- node {{\midarrow}} ({b}-west);"""


def legend(x, y, items, dx):
    out = []
    for i, (color, text) in enumerate(items):
        xi = x + i * dx
        out.append(rf"""
\fill[fill={color}, draw=black!45] ({xi},{y}) rectangle ++(0.75,0.75);
\node[anchor=west, font=\large] at ({xi + 0.95},{y + 0.37}) {{{text}}};""")
    return "".join(out)


def figure() -> str:
    a = [head()]
    a.append(r"""
\node[anchor=west, font=\LARGE] at (-1.5,10.4) {\textbf{The generator: a causal transformer that writes a peptide one residue at a time}};
\node[anchor=west, font=\Large] at (-1.5,9.1) {4,767,255 parameters $=$ embeddings 22,272 $+$ six pre-norm blocks 4,738,560 $+$ output head 5,911 $+$ final norm 512. Inference only; the checkpoint is frozen.};""")

    # Row 1 — tokens in, logits out
    a.append(box("tok", "(0,0,0)", "inputc", 34, 6, 1.0, xlabel="23", zlabel="$L$", opacity=0.7))
    a.append(cap("tok", r"\textbf{Tokens so far}\\$\langle$bos$\rangle$ then residues\\23 symbols: 20 amino acids $+$ pad/bos/eos", 6, 5.4))

    a.append(box("emb", "(tok-east)", "embc", 30, 6, 2.4, xlabel="256", zlabel="$L$", offset="(3.6,0,0)"))
    a.append(conn("tok", "emb"))
    a.append(cap("emb", r"\textbf{Token embedding}\\$23 \rightarrow 256$", 6, 3.4, dx=-0.9))

    a.append(box("pos", "(emb-east)", "embc", 30, 6, 2.4, xlabel="256", zlabel="$L$", offset="(3.2,0,0)"))
    a.append(cap("pos", r"\textbf{Position embedding}\\learned, up to 64 places", 6, 3.8, dx=1.0))
    a.append(ball("sum", "(pos-east)", r"$+$", radius=2.2, offset="(3.0,0,0)"))
    a.append(conn("emb", "pos"))
    a.append(conn("pos", "sum"))

    prev = "sum"
    for i in range(1, 7):
        name = f"blk{i}"
        off = "(3.0,0,0)" if i == 1 else "(1.1,0,0)"
        a.append(banded(name, f"({prev}-east)", "attnc", "ffnc", 30, 6, 2.2, 2.2,
                        "256", "", "$L$" if i == 6 else "", offset=off))
        a.append(conn(prev, name))
        prev = name
    a.append(r"""
\node[lab, text width=11cm] at ($(blk3-south)!0.5!(blk4-south)+(0,-1.3,0.6)$) {\textbf{6 pre-norm blocks}\\LayerNorm $\rightarrow$ causal self-attention, 8 heads $\rightarrow$ LayerNorm $\rightarrow$ GELU feed-forward $256 \rightarrow 1024 \rightarrow 256$,\\each with a residual connection. 99.4\% of the parameters.};""")

    a.append(box("norm", f"({prev}-east)", "normc", 30, 6, 1.0, zlabel="$L$", offset="(3.4,0,0)", opacity=0.75))
    a.append(conn(prev, "norm"))
    a.append(cap("norm", r"\textbf{LayerNorm}", 6, 2.6, dx=-0.5))

    a.append(box("out", "(norm-east)", "headc", 30, 6, 1.2, xlabel="23", zlabel="$L$", offset="(2.6,0,0)", opacity=0.85))
    a.append(conn("norm", "out"))
    a.append(cap("out", r"\textbf{Output head}\\$256 \rightarrow 23$", 6, 3.2, dx=0.6))
    a.append(r"""
\node[note, anchor=west] at ($(out-east)+(2.0,0.9,0)$) {one score per symbol,};
\node[note, anchor=west] at ($(out-east)+(2.0,-0.3,0)$) {at every place in the sequence};""")

    # The causal mask, drawn as a lower-triangular grid
    a.append(r"""
\node[anchor=west, font=\Large] at (-1.5,-9.0) {\textbf{What makes it autoregressive}};
\node[anchor=west, font=\large, text width=15cm, align=left] at (-1.5,-12.2)
  {A causal mask lets each place attend only to the places at or before it, so the score at place $i$ never sees residue $i{+}1$.
   Sampling then runs left to right: draw a symbol from the softmax at the last place, append it, feed the longer sequence back in,
   and stop at $\langle$eos$\rangle$ or at length 50.};""")
    for r in range(6):
        for c in range(6):
            fill = "attnc" if c <= r else "white"
            a.append(rf"""
\fill[fill={fill}, draw=black!35] ({19.0 + c * 0.8},{-9.9 - r * 0.8}) rectangle ++(0.8,-0.8);""")
    a.append(r"""
\node[font=\large, anchor=south] at (21.4,-9.6) {attends to $\rightarrow$};
\node[font=\large, anchor=east, rotate=90, anchor=south] at (18.6,-12.9) {place $\rightarrow$};""")

    a.append(legend(-1.5, -16.4, [
        ("inputc", "token ids"),
        ("embc", "embedding"),
        ("attnc", "causal self-attention"),
        ("ffnc", "feed-forward"),
        ("normc", "LayerNorm"),
        ("headc", "output head"),
    ], dx=7.2))
    a.append(end())
    return "".join(a)


def build(name: str, tex: str) -> None:
    """Compile in a temporary directory so the .tex (which embeds the local PlotNeuralNet path) is not kept."""
    import tempfile

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / f"{name}.tex").write_text(tex)
        r = subprocess.run(["tectonic", "-X", "compile", f"{name}.tex"], cwd=tmp,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"tectonic failed for {name}")
        shutil.copy(tmp / f"{name}.pdf", OUT / f"{name}.pdf")
        subprocess.run(["qlmanage", "-t", "-s", "3000", "-o", str(tmp), str(tmp / f"{name}.pdf")], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(tmp / f"{name}.pdf.png", OUT / f"{name}.png")
    print("wrote", (OUT / f"{name}.pdf").relative_to(ROOT), (OUT / f"{name}.png").relative_to(ROOT))


def main() -> int:
    if not (PLOTNN / "layers" / "init.tex").is_file():
        print("Set PLOTNN_DIR to a PlotNeuralNet clone (https://github.com/HarisIqbal88/PlotNeuralNet)", file=sys.stderr)
        return 1
    build("generator", figure())
    return 0


if __name__ == "__main__":
    sys.exit(main())
