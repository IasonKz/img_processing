"""Manual TOP/BOTTOM geometry annotator for camera/module images.

Run with:  python camera_geometry_annotator.py

Pick one TOP image and one BOTTOM image. Draw labelled polygons with clicks,
choose Point for a center marker or Circle for a circular feature, and export a
separate ZIP package for each side. Coordinates are saved in source pixels,
normalized image coordinates, and (when a four-corner die_outline is marked)
normalized coordinates in a rectified die coordinate frame.

Requirements: Python 3.10+, tkinter, Pillow, numpy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

APP_TITLE = "Camera Geometry Annotator"
APP_VERSION = "1.0"
SIDES = ("TOP", "BOTTOM")
SHAPES = ("Polygon ROI", "Point / center", "Circle")
PALETTE = ["#00e5ff", "#ffe600", "#ff55ff", "#4cff4c", "#ff8c42", "#ffffff", "#5da9ff"]
HELP_TEXT = (
    "Suggested order: first draw die_outline by clicking its 4 outer corners. "
    "The tool orders those corners automatically. Then mark inner_octagon, "
    "label, corner_TL/TR/BL/BR and/or camera_center. Each ROI can have its own name and note."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] -
                   points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points))) / 2.0)


def order_quad_clockwise(points):
    """Return four corners in image-space TL, TR, BR, BL order.

    The manual annotator may receive clicks in any order. Sorting around the
    centroid and rotating to the upper-left corner avoids invalid homographies
    when the user traces down the left side first.
    """
    pts = np.asarray(points, np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    return [[float(x), float(y)] for x, y in ordered]


def homography_from_quad(quad):
    """Map source points to die-local unit square; quad order TL,TR,BR,BL."""
    targets = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    A, b = [], []
    for (x, y), (u, v) in zip(quad, targets):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y]); b.append(v)
    try:
        h = np.linalg.solve(np.asarray(A, np.float64), np.asarray(b, np.float64))
    except np.linalg.LinAlgError:
        return None
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.]], np.float64)


def transform_point(point, matrix):
    p = np.array([point[0], point[1], 1.0], np.float64)
    q = matrix @ p
    if abs(q[2]) < 1e-12:
        return None
    return [float(q[0] / q[2]), float(q[1] / q[2])]


class GeometryAnnotator:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1450x940")
        self.root.minsize(980, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.side_index = 0
        self.image_path: Path | None = None
        self.image: Image.Image | None = None
        self.image_sha = ""
        self.image_width = self.image_height = 0
        self.output_dir: Path | None = None
        self.rois: list[dict] = []
        self.current_points: list[tuple[float, float]] = []
        self.base_scale = 1.0
        self.zoom = 1.0
        self.tk_image = None
        self.canvas_image_id = None
        self._resize_job = None
        self._build_ui()
        self.root.withdraw()
        self.root.after(100, self.start)

    @property
    def side(self):
        return SIDES[self.side_index]

    def _build_ui(self):
        self.header = ttk.Frame(self.root, padding=(8, 7))
        self.header.pack(side="top", fill="x")
        self.side_label = ttk.Label(self.header, text="TOP image", font=("TkDefaultFont", 13, "bold"))
        self.side_label.pack(side="left", padx=(0, 12))
        self.image_label = ttk.Label(self.header, text="No image selected", width=50)
        self.image_label.pack(side="left", padx=4)
        ttk.Button(self.header, text="Choose / replace image", command=self.choose_image).pack(side="right", padx=3)
        ttk.Button(self.header, text="Choose export folder", command=self.choose_output).pack(side="right", padx=3)

        self.help_label = ttk.Label(self.root, text=HELP_TEXT, wraplength=1380, justify="left", padding=(10, 3))
        self.help_label.pack(side="top", fill="x")

        controls = ttk.Frame(self.root, padding=(8, 6))
        controls.pack(side="top", fill="x")
        ttk.Label(controls, text="ROI name / characterization:").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar(value="die_outline")
        self.name_entry = ttk.Entry(controls, textvariable=self.name_var, width=23)
        self.name_entry.grid(row=0, column=1, sticky="w", padx=(5, 12))
        ttk.Label(controls, text="Shape:").grid(row=0, column=2, sticky="w")
        self.shape_var = tk.StringVar(value=SHAPES[0])
        self.shape_box = ttk.Combobox(controls, textvariable=self.shape_var, values=SHAPES, width=17, state="readonly")
        self.shape_box.grid(row=0, column=3, sticky="w", padx=(5, 12))
        self.shape_box.bind("<<ComboboxSelected>>", self.shape_changed)
        ttk.Label(controls, text="Note:").grid(row=0, column=4, sticky="w")
        self.note_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.note_var, width=38).grid(row=0, column=5, sticky="w", padx=(5, 8))
        ttk.Button(controls, text="Save ROI", command=self.save_roi).grid(row=0, column=6, padx=3)
        ttk.Button(controls, text="Undo point", command=self.undo_point).grid(row=0, column=7, padx=3)
        ttk.Button(controls, text="Clear drawing", command=self.clear_drawing).grid(row=0, column=8, padx=3)
        ttk.Button(controls, text="Undo saved ROI", command=self.undo_roi).grid(row=0, column=9, padx=3)

        viewer = ttk.Frame(self.root, padding=(8, 2, 8, 8))
        viewer.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(viewer, background="#252525", highlightthickness=1, highlightbackground="#777")
        self.hbar = ttk.Scrollbar(viewer, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(viewer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        viewer.rowconfigure(0, weight=1); viewer.columnconfigure(0, weight=1)
        roi_panel = ttk.Frame(viewer, padding=(9, 2, 2, 2))
        roi_panel.grid(row=0, column=2, rowspan=2, sticky="ns")
        ttk.Label(roi_panel, text="Saved ROIs", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 5))
        self.roi_list = tk.Listbox(roi_panel, width=30, height=18, exportselection=False)
        self.roi_list.pack(fill="y", expand=True)
        ttk.Button(roi_panel, text="Delete selected ROI", command=self.delete_selected_roi).pack(fill="x", pady=(6, 0))
        ttk.Label(roi_panel, text="Scroll to pan.\nMouse wheel / buttons to zoom.", justify="left").pack(anchor="w", pady=(12, 0))
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.adjust_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda e: self.adjust_zoom(1/1.15))
        self.root.bind("<Control-Return>", lambda e: self.save_roi())
        self.root.bind("<BackSpace>", lambda e: self.undo_point())

        footer = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        footer.pack(side="bottom", fill="x")
        ttk.Button(footer, text="− Zoom", command=lambda: self.adjust_zoom(1/1.2)).pack(side="left")
        ttk.Button(footer, text="Fit image", command=self.fit_image).pack(side="left", padx=5)
        ttk.Button(footer, text="+ Zoom", command=lambda: self.adjust_zoom(1.2)).pack(side="left")
        self.roi_count_label = ttk.Label(footer, text="Saved ROIs: 0")
        self.roi_count_label.pack(side="left", padx=18)
        self.status_label = ttk.Label(footer, text="Click to add polygon vertices. Save ROI closes the polygon.")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.save_side_button = ttk.Button(footer, text="Export TOP & choose BOTTOM", command=self.export_and_next)
        self.save_side_button.pack(side="right", padx=3)
        ttk.Button(footer, text="Export current side", command=self.export_current).pack(side="right", padx=3)

    def start(self):
        self.root.deiconify()
        if not self.choose_image(initial=True):
            self.root.destroy()
            return
        self.choose_output(initial=True)
        self.update_side_controls()

    def choose_image(self, initial=False):
        initialdir = str(self.image_path.parent) if self.image_path else str(Path.home())
        file = filedialog.askopenfilename(parent=self.root, title=f"Choose {self.side} image",
            initialdir=initialdir,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if not file:
            return False
        path = Path(file)
        try:
            with Image.open(path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            digest = sha256_file(path)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)
            return False
        if self.rois and not initial:
            keep = messagebox.askyesno("Replace image?", "Replacing this image clears its unsaved ROIs. Continue?", parent=self.root)
            if not keep:
                return False
        self.image_path = path
        self.image = image
        self.image_width, self.image_height = image.size
        self.image_sha = digest
        self.rois = []
        self.current_points = []
        self.zoom = 1.0
        self.image_label.configure(text=f"{path.name}  |  {self.image_width} × {self.image_height}")
        self.update_roi_list()
        self.root.after(100, self.fit_image)
        if initial:
            return True
        return True

    def choose_output(self, initial=False):
        initialdir = str(self.image_path.parent) if self.image_path else str(Path.home())
        selected = filedialog.askdirectory(parent=self.root, title="Choose folder for TOP/BOTTOM annotation packages",
                                           initialdir=initialdir, mustexist=True)
        if selected:
            self.output_dir = Path(selected)
        elif initial and self.image_path:
            self.output_dir = self.image_path.parent / "geometry_annotation_exports"
        if self.output_dir:
            self.status_label.configure(text=f"Exports will be written to: {self.output_dir}")

    def update_side_controls(self):
        self.side_label.configure(text=f"{self.side} image")
        if self.side_index == 0:
            self.save_side_button.configure(text="Export TOP & choose BOTTOM")
        else:
            self.save_side_button.configure(text="Export BOTTOM & finish")
        self.name_var.set("die_outline" if not any(r["name"].lower()=="die_outline" for r in self.rois) else "inner_octagon")
        self.status_label.configure(text=f"{self.side}: click vertices; use Save ROI to close/save. Ctrl+Enter also saves.")

    def shape_changed(self, _event=None):
        shape = self.shape_var.get()
        if shape == "Point / center":
            self.status_label.configure(text="Point: click its center, then press Save ROI.")
        elif shape == "Circle":
            self.status_label.configure(text="Circle: click center, then a point on the radius, then Save ROI.")
        else:
            self.status_label.configure(text="Polygon: click vertices in order, then press Save ROI to close it.")
        self.render_canvas()

    def on_canvas_resize(self, _event=None):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self.render_canvas)

    def _calculate_base_scale(self):
        if not self.image:
            return 1.0
        cw = max(200, self.canvas.winfo_width()-24)
        ch = max(200, self.canvas.winfo_height()-24)
        return min(cw/self.image_width, ch/self.image_height, 1.0)

    def fit_image(self):
        if self.image:
            self.zoom = 1.0
            self.base_scale = self._calculate_base_scale()
            self.render_canvas()
            self.canvas.xview_moveto(0); self.canvas.yview_moveto(0)

    def adjust_zoom(self, multiplier):
        if not self.image:
            return
        self.zoom = max(.25, min(8.0, self.zoom * multiplier))
        self.render_canvas()

    def on_wheel(self, event):
        self.adjust_zoom(1.15 if event.delta > 0 else 1/1.15)

    def on_click(self, event):
        if not self.image:
            return
        scale = self.base_scale * self.zoom
        x = self.canvas.canvasx(event.x) / scale
        y = self.canvas.canvasy(event.y) / scale
        if not (0 <= x < self.image_width and 0 <= y < self.image_height):
            return
        shape = self.shape_var.get()
        if shape == "Point / center" and self.current_points:
            self.current_points = []
        elif shape == "Circle" and len(self.current_points) >= 2:
            self.current_points = []
        self.current_points.append((float(x), float(y)))
        self.render_canvas()

    def undo_point(self):
        if self.current_points:
            self.current_points.pop()
            self.render_canvas()

    def clear_drawing(self):
        self.current_points = []
        self.render_canvas()

    def save_roi(self):
        if not self.image:
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("ROI name needed", "Enter a name/characterization, for example inner_octagon or label.", parent=self.root)
            return
        if any(r["name"].casefold() == name.casefold() for r in self.rois):
            messagebox.showwarning("Name already used", "Give each ROI a unique name, for example corner_TL and corner_BR.", parent=self.root)
            return
        shape = self.shape_var.get()
        expected = {"Polygon ROI": 3, "Point / center": 1, "Circle": 2}[shape]
        if len(self.current_points) < expected:
            messagebox.showwarning("More clicks needed", {
                "Polygon ROI":"A polygon needs at least 3 clicked points.",
                "Point / center":"Click one point for its center.",
                "Circle":"Click the center and then a point on its edge."
            }[shape], parent=self.root)
            return
        if name.casefold() == "die_outline" and (shape != "Polygon ROI" or len(self.current_points) != 4):
            messagebox.showwarning("die_outline requires 4 points", "Use Polygon ROI and click exactly four distinct outer die corners. Their order is corrected automatically.", parent=self.root)
            return
        points = self.current_points[:expected] if shape != "Polygon ROI" else self.current_points[:]
        if name.casefold() == "die_outline":
            points = order_quad_clockwise(points)
            if polygon_area(points) < 100:
                messagebox.showwarning("Invalid die outline", "The four corner points do not form a valid die outline.", parent=self.root)
                return
        if shape == "Polygon ROI" and polygon_area(points) < max(16.0, .000002*self.image_width*self.image_height):
            messagebox.showwarning("Polygon too small", "The polygon has almost no area. Add or reposition its vertices.", parent=self.root)
            return
        if shape == "Circle" and math.dist(points[0], points[1]) < 2:
            messagebox.showwarning("Circle too small", "Choose a second point farther from the center.", parent=self.root)
            return
        roi = {"name":name,"description":self.note_var.get().strip(),"shape":shape,
               "points":[[float(x),float(y)] for x,y in points],"color":PALETTE[len(self.rois)%len(PALETTE)]}
        self.rois.append(roi)
        self.current_points = []
        self.note_var.set("")
        if name.casefold() == "die_outline":
            self.name_var.set("inner_octagon")
        else:
            self.name_var.set("")
        self.update_roi_list()
        self.render_canvas()

    def undo_roi(self):
        if not self.rois:
            return
        self.rois.pop()
        self.update_roi_list()
        self.render_canvas()

    def update_roi_list(self):
        self.roi_count_label.configure(text=f"Saved ROIs: {len(self.rois)}")
        if hasattr(self, "roi_list"):
            self.roi_list.delete(0, "end")
            for roi in self.rois:
                self.roi_list.insert("end", f"{roi['name']}  [{roi['shape']}; {len(roi['points'])} pts]")

    def delete_selected_roi(self):
        if not self.rois or not self.roi_list.curselection():
            return
        index = int(self.roi_list.curselection()[0])
        del self.rois[index]
        self.update_roi_list()
        self.render_canvas()

    def render_canvas(self):
        if not self.image:
            return
        if self.canvas.winfo_width() < 30 or self.canvas.winfo_height() < 30:
            return
        self.base_scale = self._calculate_base_scale()
        scale = self.base_scale*self.zoom
        dw=max(1,int(round(self.image_width*scale)))
        dh=max(1,int(round(self.image_height*scale)))
        try:
            xview=self.canvas.xview()[0]; yview=self.canvas.yview()[0]
        except Exception:
            xview=yview=0.0
        resized=self.image.resize((dw,dh), Image.Resampling.LANCZOS)
        self.tk_image=ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0,0,image=self.tk_image,anchor="nw")
        self.canvas.configure(scrollregion=(0,0,dw,dh))
        self.canvas.xview_moveto(xview); self.canvas.yview_moveto(yview)
        for roi in self.rois:
            self._draw_roi_on_canvas(roi, scale, current=False)
        if self.current_points:
            temp={"name":"current","shape":self.shape_var.get(),
                  "points":[[x,y] for x,y in self.current_points],"color":"#ff4747"}
            self._draw_roi_on_canvas(temp,scale,current=True)

    def _draw_roi_on_canvas(self, roi, scale, current):
        points=[(p[0]*scale,p[1]*scale) for p in roi["points"]]
        color=roi["color"]
        width=max(2,int(round(2.0*self.zoom)))
        shape=roi["shape"]
        if shape == "Polygon ROI":
            if len(points)>=2:
                chain=points+[points[0]] if not current else points
                for a,b in zip(chain[:-1],chain[1:]):
                    self.canvas.create_line(*a,*b,fill=color,width=width)
            for i,(x,y) in enumerate(points):
                rad=max(3,int(4*self.zoom))
                self.canvas.create_oval(x-rad,y-rad,x+rad,y+rad,fill=color,outline="black")
                self.canvas.create_text(x+8,y-8,text=str(i+1),fill="white",anchor="sw",font=("TkDefaultFont",max(8,int(10*self.zoom)),"bold"))
            if not current and points:
                self.canvas.create_text(points[0][0]+6,points[0][1]+15,text=roi["name"],fill=color,anchor="nw",font=("TkDefaultFont",max(9,int(12*self.zoom)),"bold"))
        elif shape == "Point / center" and points:
            x,y=points[0]; r=max(6,int(9*self.zoom))
            self.canvas.create_line(x-r,y,x+r,y,fill=color,width=width)
            self.canvas.create_line(x,y-r,x,y+r,fill=color,width=width)
            self.canvas.create_oval(x-3,y-3,x+3,y+3,fill=color)
            self.canvas.create_text(x+10,y+10,text=roi["name"],fill=color,anchor="nw")
        elif shape == "Circle" and points:
            x,y=points[0]
            if len(points)==2:
                r=math.dist(points[0],points[1])
                self.canvas.create_oval(x-r,y-r,x+r,y+r,outline=color,width=width)
            else:
                self.canvas.create_oval(x-5,y-5,x+5,y+5,outline=color,width=width)
            if not current:
                self.canvas.create_text(x+8,y+8,text=roi["name"],fill=color,anchor="nw")

    def _annotated_preview(self):
        result=self.image.copy()
        draw=ImageDraw.Draw(result)
        try:
            font=ImageFont.truetype("DejaVuSans.ttf",max(14,int(min(self.image_width,self.image_height)*.018)))
        except Exception:
            font=ImageFont.load_default()
        for roi in self.rois:
            pts=[tuple(p) for p in roi["points"]]
            color=roi["color"]
            if roi["shape"]=="Polygon ROI":
                draw.line(pts+[pts[0]],fill=color,width=max(3,int(min(self.image.size)*.004)),joint="curve")
                for i,(x,y) in enumerate(pts):
                    rad=max(4,int(min(self.image.size)*.006))
                    draw.ellipse((x-rad,y-rad,x+rad,y+rad),fill=color,outline="black",width=2)
                    draw.text((x+rad,y-rad),str(i+1),fill="white",font=font,stroke_width=2,stroke_fill="black")
                x,y=pts[0]
            elif roi["shape"]=="Point / center":
                x,y=pts[0]; rad=max(8,int(min(self.image.size)*.012))
                draw.line((x-rad,y,x+rad,y),fill=color,width=max(3,rad//3))
                draw.line((x,y-rad,x,y+rad),fill=color,width=max(3,rad//3))
                draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
            else:
                x,y=pts[0]; rad=math.dist(pts[0],pts[1])
                draw.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=max(3,int(min(self.image.size)*.004)))
            draw.text((x+8,y+8),roi["name"],fill=color,font=font,stroke_width=3,stroke_fill="black")
        return result

    def _build_export(self):
        W,H=self.image_width,self.image_height
        die_roi=next((r for r in self.rois if r["name"].casefold()=="die_outline"),None)
        die_H=None
        if die_roi and len(die_roi["points"])==4:
            die_H=homography_from_quad(die_roi["points"])
        exports=[]
        for idx,roi in enumerate(self.rois,1):
            points=roi["points"]
            row={"roi_id":idx,"name":roi["name"],"description":roi["description"],"shape":roi["shape"],
                 "image_pixel_points":points,
                 "image_normalized_points":[[p[0]/max(W-1,1),p[1]/max(H-1,1)] for p in points],
                 "closed":roi["shape"]=="Polygon ROI",
                 "area_pixels_squared":polygon_area(points) if roi["shape"]=="Polygon ROI" else None,
                 "area_fraction_of_image":polygon_area(points)/max(W*H,1) if roi["shape"]=="Polygon ROI" else None,
                 "display_color":roi["color"]}
            if roi["shape"]=="Point / center":
                row["center_image_pixels"]=points[0]
                row["center_image_normalized"]=[points[0][0]/max(W-1,1),points[0][1]/max(H-1,1)]
            if roi["shape"]=="Circle":
                radius=math.dist(points[0],points[1])
                row["center_image_pixels"]=points[0]
                row["radius_pixels"]=radius
                row["radius_fraction_of_image_diagonal"]=radius/math.hypot(W,H)
            if die_H is not None:
                mapped=[transform_point(p,die_H) for p in points]
                row["die_frame_normalized_points"]=mapped
                if roi["shape"]=="Polygon ROI":
                    row["area_fraction_of_die_frame"]=polygon_area(mapped)
                if roi["shape"]=="Point / center":
                    row["center_die_frame_normalized"]=mapped[0]
                if roi["shape"]=="Circle":
                    center=mapped[0]
                    radius=math.dist(points[0],points[1])
                    sample=[[points[0][0]+radius*math.cos(2*math.pi*k/32),
                             points[0][1]+radius*math.sin(2*math.pi*k/32)] for k in range(32)]
                    local=[transform_point(p,die_H) for p in sample]
                    local_r=[math.dist(center,p) for p in local if p]
                    row["radius_die_frame_mean_normalized"]=float(sum(local_r)/len(local_r)) if local_r else None
            exports.append(row)
        return {
            "schema":"camera_geometry_annotation.v1",
            "app_version":APP_VERSION,
            "side":self.side,
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "source_image":{"filename":self.image_path.name,"sha256":self.image_sha,
                            "width_pixels":W,"height_pixels":H,"aspect_ratio":W/H,
                            "coordinate_origin":"top-left","x_direction":"right","y_direction":"down",
                            "exif_orientation_applied":True},
            "coordinate_notes":[
                "image_normalized_points range from 0 to 1 and are divided by image width-1 / height-1.",
                "die_frame_normalized_points map the four die_outline corners to [0,1]x[0,1].",
                "The annotator automatically orders the four die_outline corners in image-space TL,TR,BR,BL order.",
                "Image pixel coordinates refer to the EXIF-corrected image shown in the annotator."
            ],
            "die_frame_available":die_H is not None,
            "rois":exports
        }

    def export_current(self, quiet=False):
        if not self.image_path or not self.rois:
            messagebox.showwarning("Nothing to export", "Load an image and save at least one ROI first.", parent=self.root)
            return False
        if not self.output_dir:
            self.choose_output()
            if not self.output_dir:
                return False
        self.output_dir.mkdir(parents=True,exist_ok=True)
        side=self.side.lower()
        json_path=self.output_dir/f"{side}_annotations.json"
        preview_path=self.output_dir/f"{side}_annotated_preview.png"
        zip_path=self.output_dir/f"{side}_geometry_package.zip"
        data=self._build_export()
        preview=self._annotated_preview()
        with json_path.open("w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        preview.save(preview_path)
        with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
            z.write(json_path,arcname=json_path.name)
            z.write(preview_path,arcname=preview_path.name)
        if not quiet:
            self.status_label.configure(text=f"Saved {zip_path.name} ({len(self.rois)} ROIs)")
            messagebox.showinfo("Export complete", f"Saved package:\n{zip_path}\n\nIt contains the JSON coordinates and the marked-up image.", parent=self.root)
        return True

    def export_and_next(self):
        if self.current_points:
            messagebox.showwarning("Unfinished drawing", "Save the current ROI or clear its points before exporting this side.", parent=self.root)
            return
        if self.side_index == 0:
            if not self.export_current(quiet=True):
                return
            file=filedialog.askopenfilename(parent=self.root,title="Choose BOTTOM image",
                initialdir=str(self.image_path.parent if self.image_path else Path.home()),
                filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),("All files","*.*")])
            if not file:
                self.status_label.configure(text="TOP exported. Choose a BOTTOM image with ‘Choose / replace image’ when ready.")
                return
            self.side_index=1
            self.image_path=None; self.image=None; self.rois=[]; self.current_points=[]
            self.update_side_controls()
            self._load_path(Path(file))
            return
        if self.export_current(quiet=True):
            messagebox.showinfo("Both sides exported", f"TOP and BOTTOM packages are in:\n{self.output_dir}", parent=self.root)
            self.root.destroy()

    def _load_path(self,path):
        try:
            with Image.open(path) as im:
                image=ImageOps.exif_transpose(im).convert("RGB")
            digest=sha256_file(path)
        except Exception as exc:
            messagebox.showerror("Could not open image",str(exc),parent=self.root)
            return False
        self.image_path=path; self.image=image; self.image_sha=digest
        self.image_width,self.image_height=image.size
        self.image_label.configure(text=f"{path.name}  |  {self.image_width} × {self.image_height}")
        self.rois=[]; self.current_points=[]; self.zoom=1.0
        self.update_roi_list(); self.root.after(100,self.fit_image)
        return True

    def on_close(self):
        if self.rois:
            if messagebox.askyesno("Close annotator?", "Export the current side before closing?", parent=self.root):
                if not self.export_current(quiet=True):
                    return
        self.root.destroy()


def main():
    root=tk.Tk()
    app=GeometryAnnotator(root)
    root.mainloop()


if __name__=="__main__":
    main()
