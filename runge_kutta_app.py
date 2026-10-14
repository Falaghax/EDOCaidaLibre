"""
Método de Runge-Kutta  ·  Resuelve EDOs de la forma y' = f(x, y)
Incluye el modelo de caída con resistencia:  m v' = m g - k v

Requisitos:  pip install numpy matplotlib
Ejecutar:    python runge_kutta_app.py
"""

import math
import numpy as np

# ───────────────────────── LÓGICA NUMÉRICA ─────────────────────────

SAFE_NAMES = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
SAFE_NAMES.update({"abs": abs, "min": min, "max": max, "pow": pow, "e": math.e, "pi": math.pi})


def build_function(expr: str, extra: dict | None = None):
    """Convierte el texto 'g - (k/m)*y' en una función f(x, y)."""
    expr = expr.strip().replace("^", "**").replace(",", ".")
    if not expr:
        raise ValueError("La ecuación está vacía.")
    namespace = dict(SAFE_NAMES)
    if extra:
        namespace.update(extra)
    code = compile(expr, "<ecuacion>", "eval")
    for name in code.co_names:
        if name not in namespace and name not in ("x", "y", "v", "t"):
            raise ValueError(f"Símbolo desconocido: '{name}'")

    def f(x, y):
        local = dict(namespace)
        local.update({"x": x, "t": x, "y": y, "v": y})
        return eval(code, {"__builtins__": {}}, local)

    f(0.0, 1.0)  # prueba rápida para detectar errores
    return f


def step_rk1(f, x, y, h):  # Euler
    return y + h * f(x, y)


def step_rk2(f, x, y, h):  # Heun (trapecio)
    k1 = f(x, y)
    k2 = f(x + h, y + h * k1)
    return y + h * (k1 + k2) / 2


def step_rk4(f, x, y, h):  # Runge-Kutta clásico
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


STEPPERS = {"RK1": step_rk1, "RK2": step_rk2, "RK4": step_rk4}


def solve(f, x0, y0, x_end, h, method):
    """Devuelve listas (xs, ys) aplicando el método hasta x_end."""
    if h <= 0:
        raise ValueError("El paso h debe ser mayor que 0.")
    if x_end == x0:
        raise ValueError("x debe ser distinto de x0.")
    direction = 1 if x_end > x0 else -1
    n = int(round(abs(x_end - x0) / h))
    if n < 1:
        raise ValueError("El paso h es mayor que el intervalo.")
    if n > 100000:
        raise ValueError("Demasiados pasos; aumenta h.")
    hh = direction * h
    step = STEPPERS[method]
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for i in range(1, n + 1):
        y = step(f, x, y, hh)
        x = x0 + i * hh
        xs.append(x)
        ys.append(y)
        if not math.isfinite(y):
            raise ValueError("La solución diverge (valor infinito).")
    return xs, ys


def reference_curve(f, x0, y0, xa, xb, n=500):
    """Curva suave de referencia (RK4 con paso muy fino) en [xa, xb]."""
    xa, xb = min(xa, xb), max(xa, xb)

    def march(x_start, x_stop):
        if x_stop == x_start:
            return [x_start], [y0]
        hh = (x_stop - x_start) / n
        xs, ys, x, y = [x_start], [y0], x_start, y0
        for i in range(1, n + 1):
            y = step_rk4(f, x, y, hh)
            x = x_start + i * hh
            xs.append(x)
            ys.append(y)
            if not math.isfinite(y) or abs(y) > 1e12:
                break
        return xs, ys

    fx, fy = march(x0, xb)
    bx, by = march(x0, xa)
    return bx[::-1][:-1] + fx, by[::-1][:-1] + fy


# ───────────────────────────── INTERFAZ ─────────────────────────────

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    # Paleta
    BG = "#0b1020"
    PANEL = "#141b34"
    PANEL2 = "#1c2547"
    TEXT = "#e8ecff"
    MUTED = "#8b95c9"
    CYAN = "#22d3ee"
    PINK = "#f472b6"
    AMBER = "#fbbf24"
    LIME = "#a3e635"
    VIOLET = "#8b5cf6"

    METHOD_COLORS = {"RK1": AMBER, "RK2": LIME, "RK4": PINK}

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Método de Runge-Kutta")
            self.geometry("1280x760")
            self.minsize(1100, 680)
            self.configure(bg=BG)
            self._styles()
            self._header()
            self._body()
            self.calcular()

        # ---------- estilos ----------
        def _styles(self):
            s = ttk.Style(self)
            s.theme_use("clam")
            s.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, rowheight=30, borderwidth=0,
                        font=("Consolas", 11))
            s.configure("Treeview.Heading", background=VIOLET, foreground="white",
                        font=("Segoe UI", 11, "bold"), relief="flat", padding=6)
            s.map("Treeview", background=[("selected", PANEL2)])
            s.map("Treeview.Heading", background=[("active", "#7c3aed")])

        # ---------- encabezado ----------
        def _header(self):
            head = tk.Canvas(self, height=76, bg=BG, highlightthickness=0)
            head.pack(fill="x")
            # degradado horizontal
            w = 3000
            c1, c2 = (0x6d, 0x28, 0xd9), (0x06, 0xb6, 0xd4)
            for i in range(0, w, 6):
                t = i / w
                col = "#%02x%02x%02x" % tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
                head.create_rectangle(i, 0, i + 6, 76, fill=col, outline=col)
            head.create_text(28, 28, text="⚡ Runge-Kutta", anchor="w",
                             fill="white", font=("Segoe UI", 24, "bold"))
            head.create_text(30, 58, text="Modelo de caída con resistencia:   m·v′ = m·g − k·v",
                             anchor="w", fill="#e0e7ff", font=("Segoe UI", 11))

        # ---------- cuerpo ----------
        def _body(self):
            body = tk.Frame(self, bg=BG)
            body.pack(fill="both", expand=True, padx=16, pady=14)
            body.columnconfigure(1, weight=1)
            body.rowconfigure(0, weight=1)

            self._left(body)
            self._center(body)
            self._right(body)

        def _card(self, parent, **grid):
            f = tk.Frame(parent, bg=PANEL, highlightbackground=PANEL2, highlightthickness=1)
            f.grid(**grid)
            return f

        def _entry(self, parent, var, width=10, font=("Consolas", 13)):
            return tk.Entry(parent, textvariable=var, width=width, font=font,
                            bg=PANEL2, fg=CYAN, insertbackground=CYAN,
                            relief="flat", highlightthickness=2,
                            highlightbackground=PANEL2, highlightcolor=CYAN,
                            justify="center")

        def _label(self, parent, text, size=10, color=MUTED, bold=False):
            return tk.Label(parent, text=text, bg=PANEL, fg=color,
                            font=("Segoe UI", size, "bold" if bold else "normal"))

        # panel izquierdo: ecuación + tabla
        def _left(self, body):
            card = self._card(body, row=0, column=0, sticky="ns", padx=(0, 12))
            card.configure(width=440)

            self._label(card, "ECUACIÓN DIFERENCIAL", 10, CYAN, True).pack(anchor="w", padx=16, pady=(14, 4))
            row = tk.Frame(card, bg=PANEL)
            row.pack(fill="x", padx=16)
            tk.Label(row, text="y′ =", bg=PANEL, fg=PINK,
                     font=("Consolas", 16, "bold")).pack(side="left")
            self.var_eq = tk.StringVar(value="9.8 - 0.5*y")
            self._entry(row, self.var_eq, width=28).pack(side="left", padx=8, ipady=6, fill="x", expand=True)
            self._label(card, "Variables: x (o t) y y (o v).  Ej: 2*y-6,  sin(x)-y",
                        9).pack(anchor="w", padx=16, pady=(4, 8))

            # Asistente del modelo m v' = m g - k v
            box = tk.Frame(card, bg=PANEL2)
            box.pack(fill="x", padx=16, pady=(0, 10))
            tk.Label(box, text="Asistente  m·v′ = m·g − k·v", bg=PANEL2, fg=AMBER,
                     font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=7,
                                                         sticky="w", padx=10, pady=(8, 4))
            self.var_m = tk.StringVar(value="2")
            self.var_g = tk.StringVar(value="9.8")
            self.var_k = tk.StringVar(value="1")
            for i, (lbl, var) in enumerate((("m", self.var_m), ("g", self.var_g), ("k", self.var_k))):
                tk.Label(box, text=lbl + "=", bg=PANEL2, fg=TEXT,
                         font=("Consolas", 12, "bold")).grid(row=1, column=i * 2, padx=(10, 2), pady=(0, 10))
                self._entry(box, var, width=5, font=("Consolas", 12)).grid(row=1, column=i * 2 + 1, ipady=3, pady=(0, 10))
            tk.Button(box, text="Usar", command=self.usar_modelo, bg=VIOLET, fg="white",
                      activebackground="#7c3aed", activeforeground="white",
                      relief="flat", font=("Segoe UI", 10, "bold"),
                      cursor="hand2", padx=12).grid(row=1, column=6, padx=10, pady=(0, 10))

            self._label(card, "TABLA DE RESULTADOS", 10, CYAN, True).pack(anchor="w", padx=16, pady=(4, 4))
            frame = tk.Frame(card, bg=PANEL)
            frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
            cols = ("xn", "rk1", "rk2", "rk4")
            self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=14)
            for c, t in zip(cols, ("Xn", "RK1", "RK2", "RK4")):
                self.tree.heading(c, text=t)
                self.tree.column(c, width=86, anchor="center")
            sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            self.tree.tag_configure("odd", background="#18203d")
            self.tree.tag_configure("even", background=PANEL)

        # panel central: gráfica
        def _center(self, body):
            card = self._card(body, row=0, column=1, sticky="nsew", padx=(0, 12))
            self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=PANEL)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=card)
            self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))
            bar = tk.Frame(card, bg=PANEL)
            bar.pack(fill="x", padx=8, pady=(0, 6))
            tb = NavigationToolbar2Tk(self.canvas, bar)
            tb.config(background=PANEL)
            for w in tb.winfo_children():
                try:
                    w.config(background=PANEL)
                except tk.TclError:
                    pass
            tb.update()

        # panel derecho: parámetros
        def _right(self, body):
            card = self._card(body, row=0, column=2, sticky="ns")
            card.configure(width=250)
            inner = tk.Frame(card, bg=PANEL)
            inner.pack(padx=22, pady=16)

            self._label(inner, "PARÁMETROS", 10, CYAN, True).pack(anchor="w", pady=(0, 10))

            self.var_x0 = tk.StringVar(value="0")
            self.var_y0 = tk.StringVar(value="0")
            self.var_x = tk.StringVar(value="5")
            self.var_h = tk.StringVar(value="0.5")
            for text, sub, var in (("Valor inicial x₀", "x0", self.var_x0),
                                   ("Valor inicial y₀ (v₀)", "y0", self.var_y0),
                                   ("Valor a calcular y(x)", "x", self.var_x),
                                   ("Valor del paso h", "h", self.var_h)):
                self._label(inner, text, 10, TEXT).pack(anchor="w", pady=(8, 2))
                r = tk.Frame(inner, bg=PANEL)
                r.pack(fill="x")
                tk.Label(r, text=sub + " =", bg=PANEL, fg=PINK, width=4,
                         font=("Consolas", 12, "bold")).pack(side="left")
                self._entry(r, var, width=10).pack(side="left", ipady=4, padx=4)

            self._label(inner, "Aproximación a mostrar", 10, TEXT).pack(anchor="w", pady=(18, 4))
            self.var_m_sel = tk.StringVar(value="RK4")
            for m in ("RK1", "RK2", "RK4", "Todos"):
                tk.Radiobutton(inner, text=m if m != "Todos" else "Todos (comparar)",
                               variable=self.var_m_sel, value=m, command=self.calcular,
                               bg=PANEL, fg=METHOD_COLORS.get(m, CYAN), selectcolor=PANEL2,
                               activebackground=PANEL, activeforeground=TEXT,
                               font=("Segoe UI", 11, "bold"), anchor="w",
                               highlightthickness=0).pack(fill="x", pady=1)

            self.lbl_result = tk.Label(inner, text="", bg=PANEL2, fg=LIME, wraplength=200,
                                       font=("Consolas", 11, "bold"), pady=8)
            self.lbl_result.pack(fill="x", pady=(16, 8))

            tk.Button(inner, text="▶  CALCULAR", command=self.calcular, bg=CYAN, fg=BG,
                      activebackground="#67e8f9", relief="flat", cursor="hand2",
                      font=("Segoe UI", 12, "bold"), pady=8).pack(fill="x", pady=(4, 8))
            tk.Button(inner, text="✕  Cerrar", command=self.destroy, bg=PINK, fg=BG,
                      activebackground="#f9a8d4", relief="flat", cursor="hand2",
                      font=("Segoe UI", 12, "bold"), pady=8).pack(fill="x")

        # ---------- acciones ----------
        def usar_modelo(self):
            try:
                m = float(self.var_m.get().replace(",", "."))
                g = float(self.var_g.get().replace(",", "."))
                k = float(self.var_k.get().replace(",", "."))
                if m == 0:
                    raise ValueError("m no puede ser 0.")
            except ValueError as e:
                messagebox.showerror("Modelo", f"Valores inválidos: {e}")
                return
            # m v' = m g - k v   ->   v' = g - (k/m) v
            self.var_eq.set(f"{g:g} - ({k:g}/{m:g})*y")
            self.calcular()

        def _num(self, var, name):
            try:
                return float(var.get().replace(",", "."))
            except ValueError:
                raise ValueError(f"'{name}' debe ser un número.")

        def calcular(self):
            try:
                f = build_function(self.var_eq.get())
                x0 = self._num(self.var_x0, "x0")
                y0 = self._num(self.var_y0, "y0")
                xf = self._num(self.var_x, "x")
                h = self._num(self.var_h, "h")
                results = {m: solve(f, x0, y0, xf, h, m) for m in STEPPERS}
            except (ValueError, SyntaxError, NameError, ZeroDivisionError, OverflowError, TypeError) as e:
                messagebox.showerror("Error", str(e))
                return

            # tabla
            self.tree.delete(*self.tree.get_children())
            xs = results["RK1"][0]
            for i, x in enumerate(xs):
                self.tree.insert("", "end", tags=("odd" if i % 2 else "even",),
                                 values=(f"{x:.4f}",
                                         f"{results['RK1'][1][i]:.4f}",
                                         f"{results['RK2'][1][i]:.4f}",
                                         f"{results['RK4'][1][i]:.4f}"))

            sel = self.var_m_sel.get()
            shown = list(STEPPERS) if sel == "Todos" else [sel]

            # gráfica
            span = abs(xf - x0)
            xa, xb = min(x0, xf) - 0.1 * span, max(x0, xf) + 0.1 * span
            cx, cy = reference_curve(f, x0, y0, xa, xb)

            ax = self.ax
            ax.clear()
            ax.set_facecolor("#0e1530")
            ax.grid(True, color="#26305a", linestyle="--", linewidth=0.7, alpha=0.8)
            ax.plot(cx, cy, color=CYAN, lw=2.6, label="Referencia (RK4 fino)", zorder=2)
            for m in shown:
                xs_m, ys_m = results[m]
                col = METHOD_COLORS[m]
                ax.plot(xs_m, ys_m, "-", color=col, lw=1.6, alpha=0.9, zorder=3)
                ax.scatter(xs_m, ys_m, s=55, color=col, edgecolors="white",
                           linewidths=1.2, zorder=4, label=f"{m}")
            ax.set_title("Solución de  y′ = " + self.var_eq.get(), color=TEXT,
                         fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("x  (tiempo)", color=MUTED)
            ax.set_ylabel("y  (velocidad)", color=MUTED)
            ax.tick_params(colors=MUTED)
            for sp in ax.spines.values():
                sp.set_color("#26305a")
            leg = ax.legend(facecolor=PANEL2, edgecolor="#26305a", labelcolor=TEXT, loc="best")
            leg.get_frame().set_alpha(0.9)
            self.fig.tight_layout()
            self.canvas.draw()

            last = shown[-1]
            self.lbl_result.config(text=f"y({xf:g}) ≈ {results[last][1][-1]:.4f}\n({last})")

    App().mainloop()
