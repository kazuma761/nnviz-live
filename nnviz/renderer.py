"""
nnviz.renderer
--------------
All pygame drawing is isolated here. Nothing in this file knows about
PyTorch, datasets, or training — it only knows how to draw things.
"""

import math
import time
import pygame
import numpy as np
from collections import deque


# ── colour palette ──────────────────────────────────────────────────────────
BG           = (8,   10,  18)
PANEL_BG     = (14,  16,  28)
PANEL_BORDER = (40,  45,  70)
GRID_BG      = (18,  22,  38)

NEURON_IDLE  = (50,  160, 100)
NEURON_RING  = (70,  210, 130)
NEURON_HOT   = (255,  70,  70)
NEURON_GLOW  = (255, 180,  60)

TEXT_MAIN    = (220, 220, 235)
TEXT_DIM     = (100, 108, 138)
TEXT_HOT     = (255,  80,  80)
TEXT_GOOD    = (70,  220, 120)
TEXT_BAD     = (255, 100,  60)
TEXT_ACCENT  = (100, 180, 255)

WEIGHT_POS   = (60,  170, 255)
WEIGHT_NEG   = (255,  80,  50)
WEIGHT_ZERO  = (25,   28,  42)

PULSE_COLOR  = (255, 230,  80)
PIX_HIGH     = (240, 240, 255)
PIX_LOW      = (12,   14,  24)

SPARK_COLOR  = (80,  180, 255)

R_NEURON     = 9
R_GLOW_MAX   = 22


# ── helpers ──────────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _weight_color(w, alpha_scale=1.0):
    """Map normalised weight in [-1,1] → RGB."""
    magnitude = abs(w)
    if w >= 0:
        base = _lerp_color(WEIGHT_ZERO, WEIGHT_POS, magnitude)
    else:
        base = _lerp_color(WEIGHT_ZERO, WEIGHT_NEG, magnitude)
    dimmed = _lerp_color(WEIGHT_ZERO, base, min(1.0, magnitude * 2.5) * alpha_scale)
    return dimmed


def _pixel_color(v):
    v = max(0.0, min(1.0, float(v)))
    return _lerp_color(PIX_LOW, PIX_HIGH, v)


def _draw_glow_circle(surface, center, radius, color, alpha):
    """Draw a soft radial glow using concentric circles."""
    if alpha <= 0 or radius <= 0:
        return
    ix, iy = int(center[0]), int(center[1])
    steps = max(4, min(12, radius // 2))
    for i in range(steps, 0, -1):
        t   = i / steps
        r   = int(radius * t)
        a   = int(alpha * (1 - t) * 0.6)
        if a <= 0 or r <= 0:
            continue
        col  = (*color, a)
        surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, col, (r + 1, r + 1), r)
        surface.blit(surf, (ix - r, iy - r),
                     special_flags=pygame.BLEND_RGBA_ADD)


def _draw_rounded_rect(surface, color, rect, radius=8,
                        border=0, border_color=None):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surface, border_color, rect, border,
                         border_radius=radius)


# ── Signal pulse ──────────────────────────────────────────────────────────────

class SignalPulse:
    """A dot that travels from one neuron to the next along a connection."""
    __slots__ = ("src", "dst", "progress", "speed",
                 "color", "weight", "layer_idx")

    def __init__(self, src, dst, speed, color, weight, layer_idx):
        self.src       = src
        self.dst       = dst
        self.progress  = 0.0
        self.speed     = speed
        self.color     = color
        self.weight    = weight
        self.layer_idx = layer_idx

    def update(self, dt):
        self.progress = min(1.0, self.progress + self.speed * dt)

    @property
    def done(self):
        return self.progress >= 1.0

    @property
    def pos(self):
        t         = max(0.0, self.progress)
        sx, sy    = self.src
        dx, dy    = self.dst
        t2        = t * t * (3 - 2 * t)          # ease-in-out
        return (_lerp(sx, dx, t2), _lerp(sy, dy, t2))


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    Self-contained pygame renderer for nnviz.

    Parameters
    ----------
    width, height   : int
    layer_sizes     : list[int]   — neurons per layer
    input_shape     : (H,W)|None  — for pixel-grid panel
    class_names     : list[str]|None
    max_visible     : int         — max neurons drawn per layer
    fps             : int
    """

    def __init__(
        self,
        width        = 1440,
        height       = 840,
        layer_sizes  = None,
        input_shape  = None,
        class_names  = None,
        max_visible  = 24,
        fps          = 60,
    ):
        pygame.init()
        pygame.display.set_caption("nnviz  ·  Neural Network Visualiser")

        self.W           = width
        self.H           = height
        self.fps         = fps
        self.layer_sizes = layer_sizes or []
        self.input_shape = input_shape
        self.class_names = class_names
        self.max_visible = max_visible

        # ── fonts ────────────────────────────────────────────────────────
        def _font(size, bold=False):
            for name in ("Consolas", "Courier New", "DejaVu Sans Mono", None):
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f is not None:
                        return f
                except Exception:
                    pass
            return pygame.font.Font(None, size)

        self.font_sm = _font(16)
        self.font_md = _font(20)
        self.font_lg = _font(26, bold=True)
        self.font_xl = _font(32, bold=True)

        # ── layout constants  (MUST be set before _make_bg / _build_layout)
        self.PANEL_W = 270
        self.GRID_W  = 240
        self.NET_X0  = self.PANEL_W + self.GRID_W + 24

        # ── pygame surfaces ──────────────────────────────────────────────
        self.win      = pygame.display.set_mode((width, height),
                                                pygame.DOUBLEBUF)
        self.clock    = pygame.time.Clock()
        self._bg_surf = self._make_bg()

        # ── network layout ───────────────────────────────────────────────
        self.neuron_pos    = []
        self.connections   = []
        self._conn_weights = []
        self._weight_display = []   # smoothed copies, built in _build_layout
        self._build_layout()

        # ── animation state ──────────────────────────────────────────────
        self._pulses         = []
        self._neuron_glow    = []
        self._last_time      = time.perf_counter()
        self._anim_speed     = 1.0
        self._pulse_base_spd = 1.8
        self._pending_pulse  = False

        # ── metric history ───────────────────────────────────────────────
        self._loss_history = deque(maxlen=120)
        self._acc_history  = deque(maxlen=120)

        # ── init per-neuron glow arrays ──────────────────────────────────
        self._neuron_glow = [[0.0] * len(layer)
                             for layer in self.neuron_pos]

    # ── background ────────────────────────────────────────────────────────────

    def _make_bg(self):
        surf = pygame.Surface((self.W, self.H))
        surf.fill(BG)
        net_x0 = self.NET_X0
        for x in range(net_x0, self.W):
            t   = (x - net_x0) / max(1, self.W - net_x0)
            val = int(8 + 6 * math.sin(t * math.pi))
            pygame.draw.line(surf, (val, val + 2, val + 10),
                             (x, 0), (x, self.H))
        return surf

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_layout(self):
        """Pre-compute neuron positions, connection list, and weight arrays."""
        self.neuron_pos      = []
        self.connections     = []
        self._conn_weights   = []
        self._weight_display = []

        n = len(self.layer_sizes)
        if n == 0:
            return

        margin_x = 60
        margin_y = 60
        net_w    = self.W - self.NET_X0 - margin_x
        net_h    = self.H - margin_y * 2
        x_step   = net_w / max(n - 1, 1)

        # neuron positions
        for li, size in enumerate(self.layer_sizes):
            vis = min(size, self.max_visible)
            x   = self.NET_X0 + li * x_step
            ys  = self._evenly_spaced(vis, margin_y, margin_y + net_h)
            self.neuron_pos.append([(x, y) for y in ys])

        # connections + weight storage
        for li in range(1, n):
            prev  = self.neuron_pos[li - 1]
            curr  = self.neuron_pos[li]
            conns = []
            for p in prev:
                for c in curr:
                    w = [0.0]           # mutable weight container
                    conns.append((p, c, w))
            self.connections.append(conns)
            # smoothed display weights — same length as conns
            self._weight_display.append([0.0] * len(conns))

    @staticmethod
    def _evenly_spaced(n, y_min, y_max):
        if n == 1:
            return [(y_min + y_max) / 2]
        step = (y_max - y_min) / (n - 1)
        return [y_min + i * step for i in range(n)]

    # ── public setters ────────────────────────────────────────────────────────

    def update_weights(self, weight_matrices):
        """
        Push new weight matrices and trigger a signal-pulse wave.

        weight_matrices : list of np.ndarray, shape (out, in) per layer gap.
        """
        for li, wmat in enumerate(weight_matrices):
            if li >= len(self.connections):
                break

            n_from_vis = len(self.neuron_pos[li])
            n_to_vis   = len(self.neuron_pos[li + 1])
            n_from     = self.layer_sizes[li]
            n_to       = self.layer_sizes[li + 1]

            from_idx  = np.round(
                np.linspace(0, n_from - 1, n_from_vis)
            ).astype(int)
            to_idx    = np.round(
                np.linspace(0, n_to - 1, n_to_vis)
            ).astype(int)
            w_abs_max = float(np.abs(wmat).max()) + 1e-8

            ci = 0
            for fi in range(n_from_vis):
                for ti in range(n_to_vis):
                    raw_w = float(wmat[to_idx[ti], from_idx[fi]])
                    self.connections[li][ci][2][0] = raw_w / w_abs_max
                    ci += 1

            # make sure display array matches (safety)
            if len(self._weight_display[li]) != len(self.connections[li]):
                self._weight_display[li] = [0.0] * len(self.connections[li])

        self._pending_pulse = True

    def update_speed(self, speed_level):
        self._anim_speed = float(speed_level)

    def push_loss(self, loss, acc=None):
        if loss is not None:
            self._loss_history.append(float(loss))
        if acc is not None:
            self._acc_history.append(float(acc))

    # ── pulse system ──────────────────────────────────────────────────────────

    def _fire_pulse_wave(self):
        if not self.connections:
            return

        spd                 = self._pulse_base_spd * self._anim_speed
        MAX_PER_LAYER       = 12

        for li, layer_conns in enumerate(self.connections):
            indices = list(range(len(layer_conns)))
            if len(indices) > MAX_PER_LAYER:
                indices.sort(
                    key=lambda i: abs(layer_conns[i][2][0]),
                    reverse=True
                )
                indices = indices[:MAX_PER_LAYER]

            for ci in indices:
                p, c, w_box = layer_conns[ci]
                w   = w_box[0]
                col = WEIGHT_POS if w >= 0 else WEIGHT_NEG
                col = _lerp_color(PULSE_COLOR, col, 0.4)

                pulse          = SignalPulse(
                    src       = p,
                    dst       = c,
                    speed     = spd,
                    color     = col,
                    weight    = w,
                    layer_idx = li,
                )
                # stagger by layer so signal travels visually left-to-right
                pulse.progress = -li * 0.38
                self._pulses.append(pulse)

    def _update_pulses(self, dt):
        remaining = []
        for pulse in self._pulses:
            pulse.update(dt)
            if pulse.done:
                # light up destination neuron
                dest_li = pulse.layer_idx + 1
                if dest_li < len(self.neuron_pos):
                    best_i, best_d = 0, float("inf")
                    for ni, pos in enumerate(self.neuron_pos[dest_li]):
                        d = abs(pos[1] - pulse.dst[1])
                        if d < best_d:
                            best_d, best_i = d, ni
                    if dest_li < len(self._neuron_glow) and \
                       best_i < len(self._neuron_glow[dest_li]):
                        self._neuron_glow[dest_li][best_i] = min(
                            1.0,
                            self._neuron_glow[dest_li][best_i] + 0.9
                        )
            else:
                remaining.append(pulse)
        self._pulses = remaining

    def _decay_glows(self, dt):
        decay = 2.5 * self._anim_speed
        for layer_glows in self._neuron_glow:
            for i in range(len(layer_glows)):
                layer_glows[i] = max(0.0, layer_glows[i] - decay * dt)

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_pixel_grid(self, pixels):
        x0      = self.PANEL_W + 6
        y0      = 6
        avail_w = self.GRID_W - 12
        avail_h = self.H - 12

        _draw_rounded_rect(self.win, GRID_BG,
                           pygame.Rect(x0, y0, avail_w, avail_h), radius=6)

        if pixels is None:
            lbl = self.font_sm.render("No input yet", True, TEXT_DIM)
            self.win.blit(lbl, (x0 + 10, y0 + 10))
            return

        # reshape to 2-D grid
        if self.input_shape is not None:
            H, W = self.input_shape
            flat = pixels[: H * W]
            if len(flat) < H * W:
                flat = np.pad(flat, (0, H * W - len(flat)))
            grid = flat.reshape(H, W)
        else:
            n    = len(pixels)
            side = int(math.ceil(math.sqrt(n)))
            pad  = np.zeros(side * side, dtype=np.float32)
            pad[:n] = pixels
            grid = pad.reshape(side, side)
            H, W = side, side

        cell_w = avail_w / W
        cell_h = avail_h / H

        for row in range(H):
            for col in range(W):
                c  = _pixel_color(float(grid[row, col]))
                rx = int(x0 + col * cell_w)
                ry = int(y0 + row * cell_h)
                pygame.draw.rect(
                    self.win, c,
                    pygame.Rect(rx, ry,
                                max(1, int(cell_w) + 1),
                                max(1, int(cell_h) + 1))
                )

        pygame.draw.rect(self.win, PANEL_BORDER,
                         pygame.Rect(x0, y0, avail_w, avail_h),
                         1, border_radius=6)
        lbl = self.font_sm.render("INPUT", True, TEXT_DIM)
        self.win.blit(lbl, (x0 + 4, y0 + 4))

    def _draw_connections(self):
        """Draw weight lines with smoothly interpolated colours."""
        for li, layer_conns in enumerate(self.connections):
            # guard: ensure display array exists and has correct length
            if li >= len(self._weight_display):
                self._weight_display.append([0.0] * len(layer_conns))
            elif len(self._weight_display[li]) != len(layer_conns):
                self._weight_display[li] = [0.0] * len(layer_conns)

            disp = self._weight_display[li]

            for ci, (p, c, w_box) in enumerate(layer_conns):
                w       = w_box[0]
                dw      = _lerp(disp[ci], w, 0.12)   # smooth toward real weight
                disp[ci] = dw

                color = _weight_color(dw)
                if max(color) < 10:
                    continue
                pygame.draw.line(
                    self.win, color,
                    (int(p[0]), int(p[1])),
                    (int(c[0]), int(c[1])), 1
                )

    def _draw_pulses(self):
        for pulse in self._pulses:
            if pulse.progress < 0:
                continue
            px, py = pulse.pos
            ix, iy = int(px), int(py)

            gsurf = pygame.Surface((24, 24), pygame.SRCALPHA)
            pygame.draw.circle(gsurf, (*pulse.color, 60), (12, 12), 10)
            self.win.blit(gsurf, (ix - 12, iy - 12))

            pygame.draw.circle(self.win, pulse.color, (ix, iy), 3)
            pygame.draw.circle(self.win, (255, 255, 255), (ix, iy), 1)

    def _draw_neurons(self, predictions=None, class_names=None):
        n_layers = len(self.neuron_pos)

        for li, layer in enumerate(self.neuron_pos):
            is_output = (li == n_layers - 1)
            is_input  = (li == 0)

            # safe glow list for this layer
            if li < len(self._neuron_glow):
                glow_list = self._neuron_glow[li]
            else:
                glow_list = []

            for ni, (x, y) in enumerate(layer):
                ix, iy = int(x), int(y)
                glow   = glow_list[ni] if ni < len(glow_list) else 0.0

                # glow halo
                if glow > 0.02:
                    radius  = int(R_GLOW_MAX * glow)
                    g_color = _lerp_color(NEURON_RING, NEURON_GLOW, glow)
                    _draw_glow_circle(self.win, (ix, iy), radius,
                                      g_color, int(180 * glow))

                # output layer
               # output layer
                if is_output and predictions is not None \
                        and ni < len(predictions):
                    conf   = float(predictions[ni])
                    is_hot = (ni == int(np.argmax(predictions)))
                    ring   = NEURON_HOT if is_hot else NEURON_IDLE
                    fill_c = _lerp_color(NEURON_IDLE, NEURON_HOT, conf)

                    r_fill = max(2, int((R_NEURON - 2) * conf))
                    pygame.draw.circle(self.win, fill_c, (ix, iy), r_fill)
                    pygame.draw.circle(self.win, ring,   (ix, iy), R_NEURON, 2)

                    name = (class_names[ni]
                            if class_names and ni < len(class_names)
                            else str(ni))
                    clr  = TEXT_HOT if is_hot else TEXT_DIM

                    # keep label block above the output neuron
                    # (pushed further up + more spacing so nothing
                    # collides with the neuron circle below it)
                    bar_h   = 7
                    bar_max = 54
                    label_y = iy - R_NEURON - 48

                    name_surf = self.font_sm.render(name, True, clr)
                    name_x = max(8, min(ix - name_surf.get_width() // 2,
                                        self.W - name_surf.get_width() - 8))
                    self.win.blit(name_surf, (name_x, label_y))

                    bar_x = max(8, min(ix - bar_max // 2,
                                       self.W - bar_max - 8))
                    bar_y = label_y + 16
                    pygame.draw.rect(
                        self.win, (30, 35, 55),
                        pygame.Rect(bar_x, bar_y, bar_max, bar_h),
                        border_radius=3
                    )
                    bar_fill = max(1, int(bar_max * conf))
                    pygame.draw.rect(
                        self.win, fill_c,
                        pygame.Rect(bar_x, bar_y, bar_fill, bar_h),
                        border_radius=3
                    )

                    pct_surf = self.font_sm.render(f"{conf:.0%}", True, clr)
                    pct_x = max(8, min(ix - pct_surf.get_width() // 2,
                                       self.W - pct_surf.get_width() - 8))
                    # extra gap so the text clears the neuron ring
                    pct_y = bar_y + bar_h + 6
                    self.win.blit(pct_surf, (pct_x, pct_y))

                else:
                    # hidden / input
                    g_frac   = min(1.0, glow)
                    base_col = _lerp_color(NEURON_IDLE, NEURON_GLOW, g_frac)
                    if glow > 0.05:
                        fill_r = min(R_NEURON - 2,
                                     max(1, int((R_NEURON - 2) * glow)))
                        pygame.draw.circle(self.win, base_col,
                                           (ix, iy), fill_r)
                    pygame.draw.circle(self.win, base_col,
                                       (ix, iy), R_NEURON, 2)

                # layer label above first neuron
                # (skip it for the output column when predictions are
                # shown — the class name already labels each neuron,
                # so a generic "OUT" tag there only overlaps it)
                if ni == 0 and not (is_output and predictions is not None):
                    label = ("IN" if is_input else f"L{li}")
                    surf = self.font_sm.render(label, True, TEXT_DIM)
                    self.win.blit(surf,
                                  (ix - surf.get_width() // 2,
                                   iy - R_NEURON - 18))

    def _draw_panel(self, state):
        _draw_rounded_rect(self.win, PANEL_BG,
                           pygame.Rect(0, 0, self.PANEL_W, self.H),
                           radius=0)
        pygame.draw.line(self.win, PANEL_BORDER,
                         (self.PANEL_W - 1, 0),
                         (self.PANEL_W - 1, self.H), 1)

        pw = self.PANEL_W
        y  = 0

        def rule(yp):
            pygame.draw.line(self.win, PANEL_BORDER,
                             (10, yp), (pw - 10, yp), 1)

        def txt(text, yp, color=TEXT_MAIN, font=None, cx=False):
            f    = font or self.font_sm
            surf = f.render(str(text), True, color)
            xpos = (pw - surf.get_width()) // 2 if cx else 12
            self.win.blit(surf, (xpos, yp))

        # title
        y += 10
        txt("nnviz", y, TEXT_GOOD, self.font_xl, cx=True);  y += 34
        txt("Neural Network Visualiser", y, TEXT_DIM,
            self.font_sm, cx=True);                          y += 20
        rule(y);                                             y += 8

        # mode badge
        mode     = state.get("mode", "stopped")
        is_trn   = mode == "train"
        badge_bg = (30, 80, 45)  if is_trn else (70, 25, 25)
        badge_fg = TEXT_GOOD     if is_trn else TEXT_HOT
        badge_tx = "  ▶  TRAINING  " if is_trn else "  ■  STOPPED  "
        pygame.draw.rect(self.win, badge_bg,
                         pygame.Rect(10, y, pw - 20, 28), border_radius=5)
        pygame.draw.rect(self.win, badge_fg,
                         pygame.Rect(10, y, pw - 20, 28), 1, border_radius=5)
        bs = self.font_md.render(badge_tx, True, badge_fg)
        self.win.blit(bs, ((pw - bs.get_width()) // 2, y + 4));  y += 34

        # controls
        txt("[T]  Toggle training", y, TEXT_DIM);  y += 20
        speed  = state.get("speed", 1)
        n_bars = min(speed, 8)
        slbl   = self.font_sm.render(f"[+/-] Speed: {speed}×  ",
                                     True, TEXT_DIM)
        self.win.blit(slbl, (12, y))
        bx = 12 + slbl.get_width()
        for b in range(8):
            col = TEXT_ACCENT if b < n_bars else PANEL_BORDER
            pygame.draw.rect(self.win, col,
                             pygame.Rect(bx + b * 9, y + 3, 7, 12))
        y += 22
        rule(y);  y += 8

        # stats
        epoch    = state.get("epoch", 0)
        batch    = state.get("batch", 0)
        loss     = state.get("loss")
        acc      = state.get("accuracy")

        txt(f"Epoch   {epoch}", y);  y += 22
        txt(f"Batch   {batch}", y);  y += 22

        loss_str = f"{loss:.5f}" if loss is not None else "—"
        txt(f"Loss    {loss_str}", y);  y += 22

        if acc is not None:
            a_clr    = TEXT_GOOD if acc >= 50 else TEXT_BAD
            txt(f"Acc     {acc:.1f}%", y, a_clr);  y += 18
            bar_tot  = pw - 24
            bar_fill = int(bar_tot * acc / 100)
            pygame.draw.rect(self.win, (30, 35, 55),
                             pygame.Rect(12, y, bar_tot, 6), border_radius=3)
            pygame.draw.rect(self.win, a_clr,
                             pygame.Rect(12, y, bar_fill, 6), border_radius=3)
            y += 12
        else:
            txt("Acc     —", y, TEXT_DIM);  y += 22

        y += 4;  rule(y);  y += 8

        # prediction / target
        pred   = state.get("prediction")
        tgt    = state.get("target")
        cnames = state.get("class_names")

        def _name(v):
            if v is None:
                return "—"
            if cnames and v < len(cnames):
                return cnames[v]
            return str(v)

        correct = (pred is not None and tgt is not None and pred == tgt)
        p_clr   = (TEXT_GOOD if correct
                   else TEXT_HOT if pred is not None
                   else TEXT_DIM)

        txt("Prediction", y, TEXT_DIM);                        y += 18
        txt(f"  {_name(pred)}", y, p_clr, self.font_md);       y += 26
        txt("Target",     y, TEXT_DIM);                        y += 18
        txt(f"  {_name(tgt)}", y, TEXT_MAIN, self.font_md);    y += 26

        rule(y);  y += 8

        # loss sparkline
        if len(self._loss_history) >= 2:
            txt("Loss curve", y, TEXT_DIM);  y += 16
            sh = 40;  sw = pw - 24;  sx = 12;  sy = y
            pygame.draw.rect(self.win, (20, 22, 36),
                             pygame.Rect(sx, sy, sw, sh), border_radius=4)
            vals = list(self._loss_history)
            lo, hi = min(vals), max(vals)
            rng    = max(hi - lo, 1e-6)
            pts    = []
            for i, v in enumerate(vals):
                px_ = sx + int(i / max(len(vals) - 1, 1) * sw)
                py_ = sy + sh - int(((v - lo) / rng) * sh)
                pts.append((px_, py_))
            if len(pts) >= 2:
                pygame.draw.lines(self.win, SPARK_COLOR, False, pts, 2)
            y += sh + 6

        # architecture
        rule(y);  y += 8
        txt("Architecture", y, TEXT_DIM);  y += 18
        n_lbl = (["Input"]
                 + [f"Hidden {j}" for j in range(1, len(self.layer_sizes) - 1)]
                 + ["Output"])
        for i, s in enumerate(self.layer_sizes):
            if y > self.H - 22:
                txt("…", y, TEXT_DIM)
                break
            vis   = min(s, self.max_visible)
            shown = f"↓{vis}" if vis < s else f" {vis}"
            lname = n_lbl[i] if i < len(n_lbl) else f"L{i}"
            txt(f"  {lname}: {s}{shown}", y, TEXT_DIM);  y += 18

    # ── main draw call ────────────────────────────────────────────────────────

    def draw(self, pixels=None, predictions=None, state=None):
        """Render one complete frame."""
        now  = time.perf_counter()
        dt   = min(now - self._last_time, 0.1)
        self._last_time = now

        spd = (state or {}).get("speed", 1)
        self.update_speed(spd)

        self.push_loss(
            (state or {}).get("loss"),
            (state or {}).get("accuracy"),
        )

        if self._pending_pulse:
            self._pending_pulse = False
            # drop stale pulses that haven't moved yet to avoid pile-up
            self._pulses = [p for p in self._pulses if p.progress > 0.5]
            self._fire_pulse_wave()

        self._update_pulses(dt)
        self._decay_glows(dt)

        self.win.blit(self._bg_surf, (0, 0))
        self._draw_pixel_grid(pixels)
        self._draw_connections()
        self._draw_pulses()
        self._draw_neurons(predictions,
                           class_names=(state or {}).get("class_names"))
        self._draw_panel(state or {})

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ── events ────────────────────────────────────────────────────────────────

    def poll_events(self):
        """Return list of symbolic events."""
        out = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                out.append("quit")
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_t, pygame.K_SPACE):
                    out.append("toggle")
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS,
                               pygame.K_KP_PLUS):
                    out.append("speed_up")
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    out.append("speed_down")
                elif e.key == pygame.K_ESCAPE:
                    out.append("quit")
        return out

    def quit(self):
        pygame.quit()