"""Simple control widgets for ad-hoc simulation sessions."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle


class KnobWidget:
    """A compact circular control with drag-to-set interaction."""

    def __init__(self, ax, label: str, valmin: float, valmax: float, valinit: float = 0.0,
                 color: str = 'lightblue'):
        self.ax = ax
        self.label = label
        self.valmin = valmin
        self.valmax = valmax
        self.val = float(valinit)
        self.color = color
        self.callbacks = []
        self.pointer = None
        self.pointer_dot = None
        self.value_text = None
        self.press = False

        self._draw_knob()
        self.ax.figure.canvas.mpl_connect('button_press_event', self._on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self._on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _draw_knob(self) -> None:
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.axis('off')

        self.ax.add_patch(Circle((0, 0), 1.0, color='gray', alpha=0.3, zorder=1))
        self.ax.add_patch(Circle((0, 0), 0.8, color=self.color, alpha=0.8, zorder=2))

        # ticks
        for i in range(11):
            angle = np.pi / 2 + (i / 10) * (3 * np.pi / 2)
            r1, r2 = 1.05, 1.15
            x1, y1 = r1 * np.cos(angle), r1 * np.sin(angle)
            x2, y2 = r2 * np.cos(angle), r2 * np.sin(angle)
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, zorder=3)

        self._update_display()
        self.ax.text(0, -1.35, self.label, ha='center', va='top', fontsize=12, fontweight='bold')

    def _value_to_angle(self, value: float) -> float:
        normalized = (value - self.valmin) / (self.valmax - self.valmin)
        # 90 at minimum and 360 at maximum (clockwise sweep)
        return np.pi / 2 - normalized * (3 * np.pi / 2)

    def _angle_to_value(self, angle: float) -> float:
        normalized = (np.pi / 2 - angle) / (3 * np.pi / 2)
        return np.clip(self.valmin + normalized * (self.valmax - self.valmin), self.valmin, self.valmax)

    def _update_display(self) -> None:
        angle = self._value_to_angle(self.val)
        pointer_length = 0.6
        px, py = pointer_length * np.cos(angle), pointer_length * np.sin(angle)

        if self.pointer is None:
            self.pointer, = self.ax.plot([0, px], [0, py], 'r-', linewidth=4, zorder=4)
        else:
            self.pointer.set_data([0, px], [0, py])

        if self.pointer_dot is None:
            self.pointer_dot = Circle((px, py), 0.1, color='red', zorder=5)
            self.ax.add_patch(self.pointer_dot)
        else:
            self.pointer_dot.center = (px, py)

        if self.value_text is None:
            self.value_text = self.ax.text(0, 1.35, f'{self.val:.2f}', ha='center', va='bottom', fontsize=11,
                                          color='blue',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            self.value_text.set_text(f'{self.val:.2f}')

        self.ax.figure.canvas.draw_idle()

    def _notify_callbacks(self) -> None:
        for callback in self.callbacks:
            callback(self.val)

    def _update_from_mouse(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        angle = np.arctan2(event.ydata, event.xdata)
        new_val = self._angle_to_value(angle)
        if abs(new_val - self.val) > 1e-6:
            self.val = new_val
            self._update_display()
            self._notify_callbacks()

    def _on_press(self, event):
        if event.inaxes == self.ax:
            self.press = True
            self._update_from_mouse(event)

    def _on_release(self, event) -> None:
        self.press = False

    def _on_motion(self, event) -> None:
        if self.press and event.inaxes == self.ax:
            self._update_from_mouse(event)

    def on_changed(self, func: Callable[[float], None]) -> None:
        self.callbacks.append(func)

    def set_val(self, val: float) -> None:
        self.val = np.clip(float(val), self.valmin, self.valmax)
        self._update_display()

    def reset(self) -> None:
        self.set_val((self.valmin + self.valmax) / 2.0)


class InputControls:
    """Matplotlib control panel with steering/brake/throttle and per-wheel suspension."""

    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('Vehicle Input Controls', fontsize=16, fontweight='bold')

        self.inputs = {
            'throttle': 0.0,
            'brake': 0.0,
            'steering': 0.0,
            'suspension_fl': 0.0,
            'suspension_fr': 0.0,
            'suspension_rl': 0.0,
            'suspension_rr': 0.0,
        }
        self.callback: Optional[Callable[[Dict[str, float]], None]] = None

        self.knob_throttle = KnobWidget(
            self.fig.add_axes([0.1, 0.6, 0.2, 0.3]),
            label='Throttle',
            valmin=0.0,
            valmax=1.0,
            valinit=0.0,
            color='lightgreen',
        )
        self.knob_throttle.on_changed(lambda val: self._update_input('throttle', val))

        self.knob_brake = KnobWidget(
            self.fig.add_axes([0.4, 0.6, 0.2, 0.3]),
            label='Brake',
            valmin=0.0,
            valmax=1.0,
            valinit=0.0,
            color='salmon',
        )
        self.knob_brake.on_changed(lambda val: self._update_input('brake', val))

        self.knob_steering = KnobWidget(
            self.fig.add_axes([0.7, 0.6, 0.2, 0.3]),
            label='Steering',
            valmin=-1.0,
            valmax=1.0,
            valinit=0.0,
            color='lightblue',
        )
        self.knob_steering.on_changed(lambda val: self._update_input('steering', val))

        slider_height = 0.03
        y0, spacing = 0.45, 0.06
        slider_specs = [
            ('suspension_fl', 'Susp FL', -1.0, 1.0, 'red'),
            ('suspension_fr', 'Susp FR', -1.0, 1.0, 'blue'),
            ('suspension_rl', 'Susp RL', -1.0, 1.0, 'green'),
            ('suspension_rr', 'Susp RR', -1.0, 1.0, 'magenta'),
        ]

        self.susp_sliders = {}
        for i, (key, label, vmin, vmax, color) in enumerate(slider_specs):
            ax_slider = self.fig.add_axes([0.15, y0 - i * spacing, 0.7, slider_height])
            slider = Slider(ax_slider, label, vmin, vmax, valinit=0.0, color=color, valstep=0.01)
            slider.on_changed(lambda val, k=key: self._update_input(k, val))
            self.susp_sliders[key] = slider

        btn_reset = self.fig.add_axes([0.4, 0.05, 0.2, 0.05])
        self.btn_reset = Button(btn_reset, 'Reset All', color='lightgray', hovercolor='gray')
        self.btn_reset.on_clicked(lambda event: self._reset_all())

        ax_info = self.fig.add_axes([0.05, 0.12, 0.9, 0.05])
        ax_info.axis('off')
        self.info_text = ax_info.text(
            0.5,
            0.5,
            self._get_info_text(),
            ha='center',
            va='center',
            fontsize=11,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )

    def _update_input(self, key: str, value: float) -> None:
        self.inputs[key] = value
        self.info_text.set_text(self._get_info_text())
        self.fig.canvas.draw_idle()
        if self.callback:
            self.callback(self.inputs)

    def _get_info_text(self) -> str:
        return (
            f"Throttle: {self.inputs['throttle']:5.2f}  |  "
            f"Brake: {self.inputs['brake']:5.2f}  |  "
            f"Steering: {self.inputs['steering']:+5.2f}  |  "
            f"Susp FL={self.inputs['suspension_fl']:+5.2f}  "
            f"FR={self.inputs['suspension_fr']:+5.2f}  "
            f"RL={self.inputs['suspension_rl']:+5.2f}  "
            f"RR={self.inputs['suspension_rr']:+5.2f}"
        )

    def _reset_all(self) -> None:
        self.knob_throttle.set_val(0.0)
        self.knob_brake.set_val(0.0)
        self.knob_steering.set_val(0.0)
        for slider in self.susp_sliders.values():
            slider.set_val(0.0)
        self.inputs = {k: 0.0 for k in self.inputs}

    def on_input_changed(self, func: Callable[[Dict[str, float]], None]) -> None:
        self.callback = func

    def get_inputs(self) -> Dict[str, float]:
        return self.inputs.copy()

    def show(self) -> None:
        plt.show()


class TrackbarControls:
    """Alternative slider-only control set."""

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.fig, _ = plt.subplots(figsize=figsize)
        self.fig.suptitle('Vehicle Trackbar Controls', fontsize=14, fontweight='bold')

        self.inputs = {'throttle': 0.0, 'brake': 0.0, 'steering': 0.0, 'drive_torque': 0.0}
        self.callback: Optional[Callable[[Dict[str, float]], None]] = None
        self.sliders = {}

        slider_configs = [
            ('throttle', 'Throttle [0-1]', 0.0, 1.0, 0.0, 'green'),
            ('brake', 'Brake [0-1]', 0.0, 1.0, 0.0, 'red'),
            ('steering', 'Steering [-1,+1]', -1.0, 1.0, 0.0, 'blue'),
            ('drive_torque', 'Drive Torque [Nm]', 0.0, 500.0, 0.0, 'orange'),
        ]

        for i, (key, label, vmin, vmax, vinit, color) in enumerate(slider_configs):
            ax_slider = self.fig.add_axes([0.2, 0.8 - i * 0.15, 0.6, 0.07])
            slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit)
            slider.on_changed(lambda val, k=key: self._on_slider_change(k, val))
            self.sliders[key] = slider

        ax_reset = self.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset', color='lightgray', hovercolor='gray')
        self.btn_reset.on_clicked(lambda event: self._reset_all())

    def _on_slider_change(self, key: str, value: float) -> None:
        self.inputs[key] = value
        if self.callback:
            self.callback(self.inputs)

    def _reset_all(self) -> None:
        for slider in self.sliders.values():
            slider.set_val(slider.valmin)

    def on_input_changed(self, func: Callable[[Dict[str, float]], None]) -> None:
        self.callback = func

    def get_inputs(self) -> Dict[str, float]:
        return self.inputs.copy()

    def show(self) -> None:
        plt.show()


