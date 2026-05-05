"""
HIT137 Group Assignment 3 — Spot the Difference
================================================
Desktop application built with:
  - Python OOP  (≥3 classes, inheritance, polymorphism, encapsulation)
  - OpenCV      (image processing, 5 programmatic differences, 4 alteration types)
  - Tkinter     (side-by-side GUI, click detection, scoring, reveal)

Classes
-------
  Alteration (ABC)              ← abstract base
    ├─ ColourShiftAlteration    ← shifts HSV hue in a region
    ├─ BrightnessAlteration     ← darkens / lightens a region
    ├─ BlurAlteration           ← Gaussian blur on a region
    └─ NoiseAlteration          ← random Gaussian noise on a region
  DifferenceRegion              ← stores one hidden difference, tracks found state
  ImageProcessor                ← OpenCV logic: load, resize, inject differences
  GameState                     ← score / mistake counter per round + cumulative
  SpotTheDifferenceApp          ← Tkinter root window, wires everything together
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random
import math
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
#  ALTERATION HIERARCHY  (inheritance + polymorphism)
# ═══════════════════════════════════════════════════════════════════════════════

class Alteration(ABC):
    """Abstract base class for all image-region alterations."""

    def __init__(self, name: str):
        self._name = name           # encapsulated attribute

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def apply(self, image: np.ndarray,
              x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return a new image with the alteration applied to region (x,y,w,h)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


# ─── Concrete alterations ─────────────────────────────────────────────────────

class ColourShiftAlteration(Alteration):
    """Shift the hue of a rectangular region in HSV colour space."""

    def __init__(self):
        super().__init__("Colour Shift")

    def apply(self, image, x, y, w, h):
        result = image.copy()
        region = result[y:y + h, x:x + w]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int32)
        shift = random.choice(list(range(35, 70)) + list(range(-70, -35)))
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        result[y:y + h, x:x + w] = cv2.cvtColor(
            hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result


class BrightnessAlteration(Alteration):
    """Uniformly brighten or darken a rectangular region."""

    def __init__(self):
        super().__init__("Brightness")

    def apply(self, image, x, y, w, h):
        result = image.copy()
        region = result[y:y + h, x:x + w].astype(np.int32)
        delta = random.choice([-60, -50, 50, 60])
        result[y:y + h, x:x + w] = np.clip(region + delta, 0, 255).astype(np.uint8)
        return result


class BlurAlteration(Alteration):
    """Apply Gaussian blur to a rectangular region."""

    def __init__(self):
        super().__init__("Blur")

    def apply(self, image, x, y, w, h):
        result = image.copy()
        region = result[y:y + h, x:x + w]
        result[y:y + h, x:x + w] = cv2.GaussianBlur(region, (21, 21), 7)
        return result


class NoiseAlteration(Alteration):
    """Add Gaussian noise to a rectangular region."""

    def __init__(self):
        super().__init__("Noise")

    def apply(self, image, x, y, w, h):
        result = image.copy()
        region = result[y:y + h, x:x + w].astype(np.float32)
        noise = np.random.normal(0, 35, region.shape)
        result[y:y + h, x:x + w] = np.clip(region + noise, 0, 255).astype(np.uint8)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DIFFERENCE REGION
# ═══════════════════════════════════════════════════════════════════════════════

class DifferenceRegion:
    """
    Represents one programmatically introduced difference.

    Stores its bounding box, which alteration was applied,
    and whether the player has found it yet.
    """

    CLICK_TOLERANCE = 35          # pixels beyond the region edge that still register

    def __init__(self, x: int, y: int, w: int, h: int, alteration: Alteration):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._alteration = alteration
        self._found: bool = False

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self._x, self._y, self._w, self._h

    @property
    def center(self) -> tuple[int, int]:
        return self._x + self._w // 2, self._y + self._h // 2

    @property
    def found(self) -> bool:
        return self._found

    @property
    def alteration_name(self) -> str:
        return self._alteration.name

    # ── methods ───────────────────────────────────────────────────────────────

    def mark_found(self) -> None:
        self._found = True

    def is_hit(self, cx: int, cy: int) -> bool:
        """True when click (cx, cy) lands inside the region (+ tolerance)."""
        px, py = self.center
        half_w = self._w // 2 + self.CLICK_TOLERANCE
        half_h = self._h // 2 + self.CLICK_TOLERANCE
        return abs(cx - px) <= half_w and abs(cy - py) <= half_h

    def overlaps(self, other: "DifferenceRegion", margin: int = 25) -> bool:
        """True when this region overlaps *other* (with a safety margin)."""
        x1, y1, w1, h1 = self.bounds
        x2, y2, w2, h2 = other.bounds
        return not (
            x1 + w1 + margin < x2 or x2 + w2 + margin < x1 or
            y1 + h1 + margin < y2 or y2 + h2 + margin < y1
        )

    def __repr__(self) -> str:
        return (f"DifferenceRegion(center={self.center}, "
                f"alteration={self.alteration_name!r}, found={self._found})")


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class ImageProcessor:
    """
    Handles all OpenCV operations:
      - loading and resizing an image
      - generating exactly 5 non-overlapping differences on a clone
      - drawing circles on display copies
    """

    NUM_DIFFERENCES = 5
    MIN_SIZE = 45
    MAX_SIZE = 85
    MAX_DIM = 620          # longest side after resize

    def __init__(self):
        self._alterations: list[Alteration] = [
            ColourShiftAlteration(),
            BrightnessAlteration(),
            BlurAlteration(),
            NoiseAlteration(),
        ]
        self._original: np.ndarray | None = None
        self._modified: np.ndarray | None = None
        self._differences: list[DifferenceRegion] = []

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def original(self) -> np.ndarray | None:
        return self._original

    @property
    def modified(self) -> np.ndarray | None:
        return self._modified

    @property
    def differences(self) -> list[DifferenceRegion]:
        return self._differences

    # ── public methods ────────────────────────────────────────────────────────

    def load(self, path: str) -> bool:
        """Load image from *path*, resize, generate differences. Returns False on failure."""
        img = cv2.imread(path)
        if img is None:
            return False
        self._original = self._resize(img)
        self._generate_differences()
        return True

    def draw_marker(self,
                    region: DifferenceRegion,
                    color: tuple[int, int, int],
                    orig_canvas: np.ndarray,
                    mod_canvas: np.ndarray) -> None:
        """Draw a circle marker on both display canvases for *region*."""
        cx, cy = region.center
        _, _, w, h = region.bounds
        radius = max(w, h) // 2 + 14
        thickness = 3
        cv2.circle(orig_canvas, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(mod_canvas,  (cx, cy), radius, color, thickness, cv2.LINE_AA)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def to_photo(img: np.ndarray) -> ImageTk.PhotoImage:
        """Convert BGR ndarray → Tkinter PhotoImage."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    # ── private helpers ───────────────────────────────────────────────────────

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(self.MAX_DIM / max(w, h), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    def _generate_differences(self) -> None:
        """Create exactly 5 non-overlapping differences on a clone of the original."""
        self._differences = []
        self._modified = self._original.copy()
        img_h, img_w = self._original.shape[:2]

        attempts = 0
        while len(self._differences) < self.NUM_DIFFERENCES and attempts < 500:
            attempts += 1
            rw = random.randint(self.MIN_SIZE, self.MAX_SIZE)
            rh = random.randint(self.MIN_SIZE, self.MAX_SIZE)
            rx = random.randint(0, img_w - rw - 1)
            ry = random.randint(0, img_h - rh - 1)

            alteration = random.choice(self._alterations)
            candidate = DifferenceRegion(rx, ry, rw, rh, alteration)

            if any(candidate.overlaps(existing) for existing in self._differences):
                continue

            self._modified = alteration.apply(self._modified, rx, ry, rw, rh)
            self._differences.append(candidate)


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME STATE
# ═══════════════════════════════════════════════════════════════════════════════

class GameState:
    """
    Tracks per-round and cumulative game progress.
      - mistakes:    wrong clicks this round (max 3)
      - total_found: cumulative across all rounds
      - locked:      True when further input is blocked
    """

    MAX_MISTAKES = 3

    def __init__(self):
        self._mistakes: int = 0
        self._total_found: int = 0
        self._locked: bool = False

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def mistakes(self) -> int:
        return self._mistakes

    @property
    def total_found(self) -> int:
        return self._total_found

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def mistakes_remaining(self) -> int:
        return max(0, self.MAX_MISTAKES - self._mistakes)

    # ── methods ───────────────────────────────────────────────────────────────

    def record_find(self) -> None:
        self._total_found += 1

    def record_mistake(self) -> bool:
        """Increment mistakes. Returns True if now maxed out."""
        self._mistakes += 1
        if self._mistakes >= self.MAX_MISTAKES:
            self._locked = True
            return True
        return False

    def lock(self) -> None:
        self._locked = True

    def reset_round(self) -> None:
        self._mistakes = 0
        self._locked = False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class SpotTheDifferenceApp:
    """
    Tkinter root window — wires ImageProcessor, GameState, and the UI together.

    Layout
    ------
      ┌─ top bar ──────────────────────────────────────────────────────┐
      │  title                  [Reveal All]  [Load Image]             │
      ├─ status strip ─────────────────────────────────────────────────┤
      │  message text           Remaining  Mistakes  Total Found       │
      ├─ canvas area ──────────────────────────────────────────────────┤
      │  [ ORIGINAL ]          [ MODIFIED  ← click here ]             │
      ├─ legend ───────────────────────────────────────────────────────┤
      │  ● Found  ● Revealed   hint text                               │
      └────────────────────────────────────────────────────────────────┘
    """

    # Colours (dark navy / accent palette)
    C_BG       = "#0e1117"
    C_PANEL    = "#161b27"
    C_BORDER   = "#1f2a3c"
    C_ACCENT   = "#e94560"
    C_BLUE     = "#2d7dd2"
    C_GOLD     = "#ffd166"
    C_GREEN    = "#06d6a0"
    C_TEXT     = "#c9d1d9"
    C_DIM      = "#586069"

    FONT_TITLE  = ("Georgia", 17, "bold")
    FONT_LABEL  = ("Courier", 9, "bold")
    FONT_STATUS = ("Helvetica", 10)
    FONT_CHIP   = ("Helvetica", 10, "bold")
    FONT_BTN    = ("Helvetica", 10, "bold")

    def __init__(self, root: tk.Tk):
        self._root = root
        self._processor = ImageProcessor()
        self._state = GameState()

        # Display copies for drawing circles
        self._orig_disp: np.ndarray | None = None
        self._mod_disp:  np.ndarray | None = None

        # PhotoImage refs — must be kept alive to avoid GC
        self._orig_photo: ImageTk.PhotoImage | None = None
        self._mod_photo:  ImageTk.PhotoImage | None = None

        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        r = self._root
        r.title("Spot the Difference  —  HIT137")
        r.configure(bg=self.C_BG)
        r.resizable(True, True)

        self._build_topbar()
        self._build_statusbar()
        self._build_canvases()
        self._build_legend()

    def _build_topbar(self) -> None:
        bar = tk.Frame(self._root, bg=self.C_PANEL, pady=10)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="🔍  SPOT  THE  DIFFERENCE",
                 font=self.FONT_TITLE, fg=self.C_ACCENT,
                 bg=self.C_PANEL).pack(side=tk.LEFT, padx=20)

        btn_kw = dict(font=self.FONT_BTN, bd=0, relief=tk.FLAT,
                      padx=14, pady=7, cursor="hand2")

        self._load_btn = tk.Button(
            bar, text="📂  Load Image",
            bg=self.C_ACCENT, fg="white",
            activebackground="#c73652", activeforeground="white",
            command=self._load_image, **btn_kw)
        self._load_btn.pack(side=tk.RIGHT, padx=(8, 16))

        self._reveal_btn = tk.Button(
            bar, text="👁  Reveal All",
            bg=self.C_BORDER, fg=self.C_TEXT,
            activebackground=self.C_BLUE, activeforeground="white",
            command=self._reveal_all, state=tk.DISABLED, **btn_kw)
        self._reveal_btn.pack(side=tk.RIGHT, padx=4)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self._root, bg=self.C_PANEL,
                       pady=6, bd=0, highlightthickness=1,
                       highlightbackground=self.C_BORDER)
        bar.pack(fill=tk.X)

        self._msg_var = tk.StringVar(value="Load an image to start playing.")
        tk.Label(bar, textvariable=self._msg_var,
                 font=self.FONT_STATUS, fg=self.C_DIM,
                 bg=self.C_PANEL).pack(side=tk.LEFT, padx=16)

        chip_frame = tk.Frame(bar, bg=self.C_PANEL)
        chip_frame.pack(side=tk.RIGHT, padx=16)

        self._remaining_var = tk.StringVar(value="Remaining: –")
        self._mistakes_var  = tk.StringVar(value="Mistakes: 0 / 3")
        self._found_var     = tk.StringVar(value="Total Found: 0")

        for var, fg in [
            (self._remaining_var, self.C_GOLD),
            (self._mistakes_var,  self.C_ACCENT),
            (self._found_var,     self.C_GREEN),
        ]:
            tk.Label(chip_frame, textvariable=var,
                     font=self.FONT_CHIP, fg=fg,
                     bg=self.C_PANEL, padx=14).pack(side=tk.LEFT)

    def _build_canvases(self) -> None:
        outer = tk.Frame(self._root, bg=self.C_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        for side, label, is_active in [
            (tk.LEFT,  "ORIGINAL",              False),
            (tk.RIGHT, "MODIFIED  ← click here", True),
        ]:
            border_col = self.C_ACCENT if is_active else self.C_BORDER
            frame = tk.Frame(outer, bg=border_col, bd=2, relief=tk.FLAT)
            frame.pack(side=side, expand=True, fill=tk.BOTH,
                       padx=(0, 8) if side == tk.LEFT else (8, 0))

            header = tk.Frame(frame, bg=border_col)
            header.pack(fill=tk.X)
            tk.Label(header, text=label,
                     font=self.FONT_LABEL,
                     fg="white" if is_active else self.C_DIM,
                     bg=border_col, pady=5).pack()

            inner = tk.Frame(frame, bg=self.C_BG)
            inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

            canvas = tk.Canvas(inner, bg=self.C_BG, cursor="crosshair",
                               highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)

            if is_active:
                self._mod_canvas = canvas
                canvas.bind("<Button-1>", self._on_click)
            else:
                self._orig_canvas = canvas

    def _build_legend(self) -> None:
        bar = tk.Frame(self._root, bg=self.C_PANEL, pady=5)
        bar.pack(fill=tk.X)
        tk.Label(bar,
                 text="  🔴 Found    🔵 Revealed    Max 3 mistakes per image",
                 font=("Helvetica", 9), fg=self.C_DIM,
                 bg=self.C_PANEL).pack()

    # ─── image loading ────────────────────────────────────────────────────────

    def _load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files",   "*.*"),
            ],
        )
        if not path:
            return

        if not self._processor.load(path):
            messagebox.showerror("Load Error",
                                 "Could not read the image.\nPlease choose a JPG, PNG, or BMP file.")
            return

        self._state.reset_round()
        self._orig_disp = self._processor.original.copy()
        self._mod_disp  = self._processor.modified.copy()

        self._refresh_canvases()
        self._reveal_btn.config(state=tk.NORMAL,
                                bg=self.C_BORDER, fg=self.C_TEXT)
        self._update_status()
        self._msg_var.set("Click the MODIFIED image to find the 5 differences!")

    # ─── canvas refresh ───────────────────────────────────────────────────────

    def _refresh_canvases(self) -> None:
        if self._orig_disp is None:
            return

        self._orig_photo = ImageProcessor.to_photo(self._orig_disp)
        self._mod_photo  = ImageProcessor.to_photo(self._mod_disp)

        h, w = self._orig_disp.shape[:2]
        for canvas, photo in [
            (self._orig_canvas, self._orig_photo),
            (self._mod_canvas,  self._mod_photo),
        ]:
            canvas.config(width=w, height=h)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # ─── click handling ───────────────────────────────────────────────────────

    def _on_click(self, event: tk.Event) -> None:
        if self._processor.original is None or self._state.locked:
            return

        cx, cy = event.x, event.y

        for region in self._processor.differences:
            if region.found:
                continue
            if region.is_hit(cx, cy):
                self._register_find(region)
                return

        # Miss
        self._register_mistake()

    def _register_find(self, region: DifferenceRegion) -> None:
        region.mark_found()
        self._state.record_find()

        # Red circle = found by player
        self._processor.draw_marker(region, (0, 0, 220),
                                    self._orig_disp, self._mod_disp)
        self._refresh_canvases()
        self._update_status()

        remaining = sum(1 for d in self._processor.differences if not d.found)
        if remaining == 0:
            self._state.lock()
            self._reveal_btn.config(state=tk.DISABLED)
            messagebox.showinfo(
                "🎉 All Found!",
                f"Excellent! You spotted all 5 differences!\n\n"
                f"Total differences found this session: {self._state.total_found}\n\n"
                "Load a new image to keep playing.",
            )

    def _register_mistake(self) -> None:
        maxed = self._state.record_mistake()
        self._update_status()
        if maxed:
            self._reveal_btn.config(state=tk.DISABLED)
            found = sum(1 for d in self._processor.differences if d.found)
            messagebox.showwarning(
                "❌ Too Many Mistakes",
                f"You've used all 3 guesses!\n\n"
                f"Differences found: {found} / 5\n\n"
                "Load a new image to try again.",
            )

    # ─── reveal ───────────────────────────────────────────────────────────────

    def _reveal_all(self) -> None:
        if self._processor.original is None:
            return
        for region in self._processor.differences:
            if not region.found:
                # Blue circle = revealed
                self._processor.draw_marker(region, (220, 80, 0),
                                            self._orig_disp, self._mod_disp)
        self._state.lock()
        self._reveal_btn.config(state=tk.DISABLED)
        self._refresh_canvases()
        self._update_status()
        self._msg_var.set("Differences revealed. Load a new image to play again.")

    # ─── status update ────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        if self._processor.original is None:
            return

        remaining = sum(1 for d in self._processor.differences if not d.found)
        self._remaining_var.set(f"Remaining: {remaining}")
        self._mistakes_var.set(
            f"Mistakes: {self._state.mistakes} / {GameState.MAX_MISTAKES}")
        self._found_var.set(f"Total Found: {self._state.total_found}")

        if self._state.locked:
            self._msg_var.set(
                "Round over — load a new image to continue.")
        else:
            self._msg_var.set(
                f"Find the differences! {self._state.mistakes_remaining} mistake(s) remaining.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = SpotTheDifferenceApp(root)
    root.mainloop()
 