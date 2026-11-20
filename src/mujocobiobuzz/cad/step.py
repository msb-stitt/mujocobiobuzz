"""Read a STEP assembly's solids with OpenCASCADE, through `cadquery-ocp`.

The only module that imports OCP, so the rest of the package runs without the `cad` group. Each
solid comes back placed in the assembly's frame, in mm, with its box, volume, centroid and a
coarse triangulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.collections import Sequence_TDF_Label
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

log = logging.getLogger(__name__)

# The triangulation's chord error, mm, and angle, rad: coarse, for a look, not a collision.
LINEAR_MM = 0.8
ANGULAR_RAD = 0.6


@dataclass
class Solid:
    """One solid of the assembly, placed: its path of instance names from the root, in mm."""

    path: list[str]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    volume_mm3: float
    centroid: NDArray[np.float64]
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int64]


def _name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return str(attr.Get().ToExtString())
    return "?"


def _triangulate(shape: TopoDS_Shape) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    BRepMesh_IncrementalMesh(shape, LINEAR_MM, False, ANGULAR_RAD, True)
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            flip = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                tris.append(
                    (base + a - 1, base + c - 1, base + b - 1)
                    if flip
                    else (base + a - 1, base + b - 1, base + c - 1)
                )
        exp.Next()
    return np.array(verts, float).reshape(-1, 3), np.array(tris, np.int64).reshape(-1, 3)


def read_solids(path: Path) -> list[Solid]:
    """Each solid of the STEP at `path`, walked from its free shapes through its components."""
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise OSError(f"could not read STEP file {path}")
    reader.Transfer(doc)
    log.info("read %s", path)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    solids: list[Solid] = []

    def walk(label: TDF_Label, loc: TopLoc_Location, names: list[str]) -> None:
        if XCAFDoc_ShapeTool.IsAssembly_s(label):
            comps = Sequence_TDF_Label()
            XCAFDoc_ShapeTool.GetComponents_s(label, comps)
            for i in range(1, comps.Length() + 1):
                comp = comps.Value(i)
                ref = TDF_Label()
                XCAFDoc_ShapeTool.GetReferredShape_s(comp, ref)
                walk(
                    ref,
                    loc.Multiplied(XCAFDoc_ShapeTool.GetLocation_s(comp)),
                    [*names, _name(comp)],
                )
            return
        shape = XCAFDoc_ShapeTool.GetShape_s(label).Moved(loc)
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        lo, hi = box.CornerMin(), box.CornerMax()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        c = props.CentreOfMass()
        verts, tris = _triangulate(shape)
        solids.append(
            Solid(
                names,
                np.array([lo.X(), lo.Y(), lo.Z()]),
                np.array([hi.X(), hi.Y(), hi.Z()]),
                float(props.Mass()),
                np.array([c.X(), c.Y(), c.Z()]),
                verts,
                tris,
            )
        )

    free = Sequence_TDF_Label()
    tool.GetFreeShapes(free)
    for i in range(1, free.Length() + 1):
        walk(free.Value(i), TopLoc_Location(), [_name(free.Value(i))])
    log.info("%d solids", len(solids))
    return solids
