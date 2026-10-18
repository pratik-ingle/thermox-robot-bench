"""Fill the generated per-problem tables into report/template.html -> docs/index.html
(the GitHub Pages site).

Placeholders: <!--@facts NAME-->, <!--@cost NAME-->, <!--@time NAME-->,
<!--@verdict NAME-->, <!--@overview-->
"""
import os
import re
import json

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(os.path.dirname(HERE), "docs")

ORDER = [
    ("kf_iso", "Robot perception", "Kalman gain, isotropic camera"),
    ("kf_depth3", "Robot perception", "Kalman gain, depth noise 3×"),
    ("kf_depth10", "Robot perception", "Kalman gain, depth noise 10×"),
    ("ik", "Robot motion", "Arm inverse kinematics"),
    ("mpc", "Robot motion", "Torque MPC"),
    ("mpc_hard", "Robot motion", "Torque MPC, low torque weight"),
    ("heat8", "Physics simulation", "Heat equation, 8×8 grid"),
    ("heat16", "Physics simulation", "Heat equation, 16×16 grid"),
] + [(f"rand_d{d}_k{k}", "Random dense matrix", f"Random SPD, d={d}, κ={k}")
     for d in (16, 64, 256) for k in (10, 100, 1000)]


def matrix_changes(t):
    return "fixed" if t["n_mat"] == 1 else ("every solve" if t["seq"] else "independent")


def facts(t):
    items = [("size d", f"{t['d']}"), ("right-hand sides", f"{t['m']}"),
             ("condition κ", f"{t['cond']:.3g}"), ("solves", f"{t['n']}"),
             ("matrix changes", matrix_changes(t))]
    return "<dl class='facts'>" + "".join(
        f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items) + "</dl>"


def overview(tables):
    rows = []
    for i, (name, fam, label) in enumerate(ORDER, 1):
        t = tables[name]
        verdict = t["verdict"].replace("optimistic chip ", "")
        rows.append(f"<tr><td>P{i}</td><th><a href='#p-{name}'>{label}</a></th><td>{fam}</td>"
                    f"<td>{t['d']}</td><td>{t['m']}</td><td>{t['cond']:.3g}</td>"
                    f"<td>{t['n']}</td><td>{matrix_changes(t)}</td><td class='v'>{verdict}</td>"
                    "</tr>")
    return ("<table class='num overview'><thead><tr><th>#</th><th>Problem</th><th>Family</th>"
            "<th>d</th><th>RHS</th><th>κ</th><th>Solves</th><th>Matrix</th>"
            "<th>Verdict at 1e-4 (chip vs CPU)</th></tr></thead><tbody>" + "".join(rows) +
            "</tbody></table>")


def main():
    with open(os.path.join(SITE, "assets", "tables.json")) as fh:
        tables = json.load(fh)
    with open(os.path.join(HERE, "template.html")) as fh:
        html = fh.read()

    def sub(m):
        kind, name = m.group(1), m.group(2)
        if kind == "overview":
            return overview(tables)
        t = tables[name]
        return facts(t) if kind == "facts" else t[kind]

    html, n = re.subn(r"<!--@(facts|cost|time|verdict|overview) ?([a-z0-9_]*)-->", sub, html)
    left = re.findall(r"<!--@[^>]*-->", html)
    try:                                   # true pixel sizes, so the layout does not jump
        from PIL import Image

        def dims(m):
            w, h = Image.open(os.path.join(SITE, m.group(1))).size
            return f'src="{m.group(1)}" width="{w}" height="{h}"'
        html = re.sub(r'src="(assets/[^"]+\.(?:png|gif))" width="\d+" height="\d+"', dims, html)
    except ImportError:
        pass
    with open(os.path.join(SITE, "index.html"), "w") as fh:
        fh.write(html)
    print(f"filled {n} placeholders -> docs/index.html; unfilled: {left}")


if __name__ == "__main__":
    main()
