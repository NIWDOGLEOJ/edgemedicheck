"""
Module 6 (display half) -- the GPIO colour status light.

The Flask screen already shows the verdict, but at a pharmacy counter the
person holding the pack is usually not looking at the screen: they are looking
at the pack. So the same three-state verdict is mirrored onto an RGB LED wired
to the Raspberry Pi's GPIO header, which is visible from anywhere at the
counter and readable without reading.

    scanning   pulsing blue      work in progress, do not lift the pack yet
    GREEN      solid green       no issue detected
    YELLOW     solid amber       verify manually
    RED        blinking red      do not dispense
    error      fast magenta      the scan itself failed

Two decisions worth stating.

First, red blinks and green does not. Red/green confusion is the most common
form of colour vision deficiency, and a device whose entire safety output is a
red-versus-green distinction would be unreadable for roughly one man in twelve.
Motion is the redundant channel: the "stop" state is the only one that moves.

Second, every method here is failure-tolerant. A loose jumper wire or a GPIO
library that refuses to initialise must never take down a scan -- the light is
an accessory to the verdict, not the verdict. Errors are logged once and the
light then degrades to a no-op for the rest of the process.

Hardware
--------
One common-cathode RGB LED (or three discrete LEDs), each channel through its
own resistor:

    red   -- 220R -- BCM 17
    green -- 220R -- BCM 27
    blue  -- 220R -- BCM 22
    common cathode -- GND

Set `EMC_LIGHT_RED_PIN` / `_GREEN_PIN` / `_BLUE_PIN` (or edit
`StatusLightConfig`) to enable it. With no pins set the whole module is inert,
which is what lets the identical code run on a developer laptop.

For a common-anode LED the common leg goes to 3V3 instead and
`EMC_LIGHT_COMMON_ANODE=1` inverts every level.
"""

from __future__ import annotations

import atexit
import logging
import math
import threading
import time
from dataclasses import dataclass, replace

from .config import CONFIG, StatusLightConfig

log = logging.getLogger(__name__)

# States. The three verdict states deliberately reuse the fusion verdict
# strings, so `light.verdict(result.verdict.verdict)` needs no translation.
IDLE = "idle"
SCANNING = "scanning"
GREEN = "green"
YELLOW = "yellow"
RED = "red"
ERROR = "error"

# Animation modes.
SOLID = "solid"
BLINK = "blink"
PULSE = "pulse"

Colour = tuple[float, float, float]

C_OFF: Colour = (0.0, 0.0, 0.0)
C_GREEN: Colour = (0.0, 1.0, 0.0)
# Not (1, 1, 0). In a single RGB package the red die is the brightest of the
# three at equal current, so an even red/green mix reads as a yellowish green.
# Holding green back to ~45% is what actually looks amber to the eye.
C_AMBER: Colour = (1.0, 0.45, 0.0)
C_RED: Colour = (1.0, 0.0, 0.0)
C_BLUE: Colour = (0.0, 0.0, 1.0)
C_MAGENTA: Colour = (1.0, 0.0, 1.0)

# Animation frame interval. 25 Hz is smooth enough for a breathing pulse and
# cheap enough to be invisible next to the pipeline's own CPU cost.
_FRAME = 0.04
# A pulse dims towards this floor rather than to black, so the light never
# looks like it switched off mid-scan.
_PULSE_FLOOR = 0.12
# Without PWM a channel is only ever fully on or fully off, so any non-zero
# level lights it. Rounding at 0.5 instead would drop amber's 45% green and
# render YELLOW as pure red -- "verify this" shown as "do not dispense".
_DIGITAL_ON = 0.02


def _digital(value: float) -> bool:
    """Collapse a channel level to on/off for a non-PWM LED."""
    return value > _DIGITAL_ON


@dataclass(frozen=True)
class Pattern:
    """What one state looks like: a colour plus how it moves."""

    colour: Colour
    mode: str = SOLID
    period: float = 1.0


# --------------------------------------------------------------------------
# GPIO backends
# --------------------------------------------------------------------------


class _Backend:
    """Writes a 0.0-1.0 level to each colour channel."""

    name = "none"
    active = False

    def set(self, colour: Colour) -> None:
        pass

    def close(self) -> None:
        pass


class _NullBackend(_Backend):
    """Used off-Pi and whenever no pins are configured."""


class _RPiGPIOBackend(_Backend):  # pragma: no cover - hardware only
    """RPi.GPIO, with one hardware-timed software PWM channel per colour."""

    name = "RPi.GPIO"
    active = True

    def __init__(self, cfg: StatusLightConfig) -> None:
        import RPi.GPIO as GPIO  # type: ignore

        self.cfg = cfg
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self._pins: list[int | None] = [cfg.red_pin, cfg.green_pin, cfg.blue_pin]
        self._pwm: list[object | None] = [None, None, None]
        # "Off" is HIGH on a common-anode LED, so the initial level has to be
        # inverted too -- otherwise the light blazes white until the first
        # scan writes a real colour.
        off_level = GPIO.HIGH if cfg.common_anode else GPIO.LOW
        for i, pin in enumerate(self._pins):
            if pin is None:
                continue
            GPIO.setup(pin, GPIO.OUT, initial=off_level)
            if cfg.pwm:
                channel = GPIO.PWM(pin, cfg.pwm_frequency_hz)
                channel.start(100.0 if cfg.common_anode else 0.0)
                self._pwm[i] = channel

    def set(self, colour: Colour) -> None:
        for i, pin in enumerate(self._pins):
            if pin is None:
                continue
            value = colour[i]
            channel = self._pwm[i]
            if channel is not None:
                duty = (1.0 - value) if self.cfg.common_anode else value
                channel.ChangeDutyCycle(duty * 100.0)
            else:
                on = _digital(value)
                if self.cfg.common_anode:
                    on = not on
                self._gpio.output(pin, self._gpio.HIGH if on else self._gpio.LOW)

    def close(self) -> None:
        for channel in self._pwm:
            if channel is not None:
                try:
                    channel.stop()
                except Exception:  # noqa: BLE001
                    pass
        pins = [p for p in self._pins if p is not None]
        if pins:
            try:
                self._gpio.cleanup(pins)
            except Exception:  # noqa: BLE001
                pass


class _GpiozeroBackend(_Backend):  # pragma: no cover - hardware only
    """gpiozero fallback.

    RPi.GPIO cannot drive the Pi 5's GPIO block at all -- it imports fine and
    then fails at `setmode` with an SOC peripheral address error -- so on
    current hardware this is the backend that actually runs.
    """

    name = "gpiozero"
    active = True

    def __init__(self, cfg: StatusLightConfig) -> None:
        from gpiozero import LED, PWMLED  # type: ignore

        self._pwm = cfg.pwm
        factory = PWMLED if cfg.pwm else LED
        self._leds: list[object | None] = []
        for pin in (cfg.red_pin, cfg.green_pin, cfg.blue_pin):
            self._leds.append(
                None if pin is None
                else factory(pin, active_high=not cfg.common_anode)
            )

    def set(self, colour: Colour) -> None:
        for i, led in enumerate(self._leds):
            if led is not None:
                led.value = colour[i] if self._pwm else int(_digital(colour[i]))

    def close(self) -> None:
        for led in self._leds:
            if led is not None:
                try:
                    led.off()
                    led.close()
                except Exception:  # noqa: BLE001
                    pass


def open_backend(cfg: StatusLightConfig) -> _Backend:
    """Open the first GPIO library that will actually drive these pins."""
    if not cfg.enabled:
        return _NullBackend()

    for factory in (_RPiGPIOBackend, _GpiozeroBackend):
        try:
            backend = factory(cfg)
        except Exception as exc:  # noqa: BLE001
            log.debug("Status light backend %s unavailable: %s",
                      factory.name, exc)
            continue
        log.info(
            "Status light on BCM pins r=%s g=%s b=%s via %s",
            cfg.red_pin, cfg.green_pin, cfg.blue_pin, backend.name,
        )
        return backend

    log.warning(
        "Status light pins are configured but no GPIO library could drive "
        "them. Install RPi.GPIO or gpiozero, or unset the EMC_LIGHT_*_PIN "
        "variables to silence this."
    )
    return _NullBackend()


# --------------------------------------------------------------------------
# The light
# --------------------------------------------------------------------------


def _scale(colour: Colour, level: float) -> Colour:
    return (colour[0] * level, colour[1] * level, colour[2] * level)


class StatusLight:
    """Drives the RGB verdict light. Every method is safe to call anywhere.

    One background thread animates the blinking and pulsing states; solid
    states need no thread at all. Changing state always cancels the previous
    animation first, so states never overlap.
    """

    def __init__(
        self,
        cfg: StatusLightConfig | None = None,
        backend: _Backend | None = None,
    ) -> None:
        self.cfg = cfg or CONFIG.status_light
        self._backend = backend if backend is not None else open_backend(self.cfg)
        self._lock = threading.RLock()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._failed = False
        self.state = IDLE

    @property
    def available(self) -> bool:
        """True when a real LED is being driven."""
        return self._backend.active and not self._failed

    @property
    def backend(self) -> str:
        return self._backend.name if self.available else "none"

    # -- states ---------------------------------------------------------

    def scanning(self) -> None:
        """Show 'working'. A no-op if a scan is already being shown."""
        if self.state == SCANNING:
            return
        # Without a blue channel the pulse would be invisible, so a two-colour
        # build gets a blinking amber instead.
        pattern = (
            Pattern(C_BLUE, PULSE, 1.6) if self.cfg.blue_pin is not None
            else Pattern(C_AMBER, BLINK, 0.5)
        )
        self._apply(SCANNING, pattern)

    def verdict(self, verdict: str) -> None:
        """Show a fusion verdict: "green", "yellow" or "red"."""
        pattern = _VERDICT_PATTERNS.get(verdict)
        if pattern is None:
            log.warning("Unknown verdict for the status light: %r", verdict)
            self.error()
            return
        if verdict == RED and not self.cfg.blink_red:
            pattern = replace(pattern, mode=SOLID)
        self._apply(verdict, pattern, hold=self.cfg.verdict_hold_seconds)

    def error(self) -> None:
        """Show that the scan itself failed, which is not a verdict."""
        self._apply(ERROR, Pattern(C_MAGENTA, BLINK, 0.25),
                    hold=self.cfg.verdict_hold_seconds)

    def off(self) -> None:
        """Return to idle (dark)."""
        self._apply(IDLE, Pattern(C_OFF))

    def close(self) -> None:
        """Turn the light off and release the pins."""
        with self._lock:
            self._cancel()
            self.state = IDLE
            try:
                self._backend.set(C_OFF)
                self._backend.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Status light close failed: %s", exc)
            self._backend = _NullBackend()

    def __enter__(self) -> "StatusLight":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ------------------------------------------------------

    def _apply(self, state: str, pattern: Pattern, hold: float = 0.0) -> None:
        with self._lock:
            self._cancel()
            self.state = state
            if not self.available:
                return

            mode = pattern.mode
            if mode == PULSE and not self.cfg.pwm:
                # Without PWM there is nothing to ramp; blink instead so the
                # state still reads as "in progress".
                pattern = replace(pattern, mode=BLINK, period=0.5)
                mode = BLINK

            if mode == SOLID and hold <= 0:
                self._write(pattern.colour)
                return

            stop = threading.Event()
            self._stop = stop
            self._thread = threading.Thread(
                target=self._run,
                args=(pattern, hold, stop),
                name="emc-status-light",
                daemon=True,
            )
            self._thread.start()

    def _cancel(self) -> None:
        """Stop the running animation. Caller holds the lock."""
        if self._stop is not None:
            self._stop.set()
        thread = self._thread
        self._stop = None
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    def _run(self, pattern: Pattern, hold: float, stop: threading.Event) -> None:
        start = time.monotonic()
        deadline = (start + hold) if hold > 0 else None
        on = True

        while not stop.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break

            if pattern.mode == PULSE:
                phase = ((now - start) % pattern.period) / pattern.period
                level = _PULSE_FLOOR + (1.0 - _PULSE_FLOOR) * (
                    0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
                )
                self._write(_scale(pattern.colour, level))
                wait = _FRAME
            elif pattern.mode == BLINK:
                self._write(pattern.colour if on else C_OFF)
                on = not on
                wait = pattern.period / 2.0
            else:  # SOLID, held for a fixed time
                self._write(pattern.colour)
                wait = max(0.0, deadline - now) if deadline else 3600.0

            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
            if stop.wait(wait):
                return

        # The hold expired rather than being interrupted, so clear the light
        # and record that nothing is being shown any more.
        with self._lock:
            if self._stop is stop:
                self.state = IDLE
                self._stop = None
                self._thread = None
                self._write(C_OFF)

    def _write(self, colour: Colour) -> None:
        if self._failed:
            return
        if self.cfg.pwm and self.cfg.brightness < 1.0:
            # Brightness is a duty-cycle scale, so it only means anything on a
            # PWM backend; scaling a digital level would just cross the on/off
            # threshold and switch the LED off entirely.
            colour = _scale(colour, max(0.0, self.cfg.brightness))
        try:
            self._backend.set(
                (min(1.0, max(0.0, colour[0])),
                 min(1.0, max(0.0, colour[1])),
                 min(1.0, max(0.0, colour[2])))
            )
        except Exception as exc:  # noqa: BLE001
            # A scan must never fail because of the indicator LED.
            self._failed = True
            log.warning("Status light disabled after a GPIO write error: %s", exc)


_VERDICT_PATTERNS: dict[str, Pattern] = {
    GREEN: Pattern(C_GREEN, SOLID),
    YELLOW: Pattern(C_AMBER, SOLID),
    RED: Pattern(C_RED, BLINK, 0.6),
}


# --------------------------------------------------------------------------
# Shared instance
# --------------------------------------------------------------------------

_LIGHT: StatusLight | None = None
_LIGHT_LOCK = threading.Lock()


def get_status_light(cfg: StatusLightConfig | None = None) -> StatusLight:
    """Return the process-wide status light, opening the GPIO pins once.

    A GPIO pin cannot be claimed twice, and the Flask server serves scans from
    several threads, so every caller shares one instance. When no pins are
    configured this is an inert object, which is why the pipeline can call it
    unconditionally.
    """
    global _LIGHT
    if _LIGHT is None:
        with _LIGHT_LOCK:
            if _LIGHT is None:
                _LIGHT = StatusLight(cfg)
                # Leaving a red LED lit after Ctrl-C would claim a verdict the
                # device is no longer standing behind.
                atexit.register(_LIGHT.close)
    return _LIGHT


def reset_status_light() -> None:
    """Release the shared light. Used by tests and by `run.py light`."""
    global _LIGHT
    with _LIGHT_LOCK:
        if _LIGHT is not None:
            _LIGHT.close()
        _LIGHT = None
