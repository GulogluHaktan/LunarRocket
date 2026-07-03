from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import numpy as np

from app.config import resolve_project_path


@dataclass
class RocketState:
    prim_path: str
    position: tuple[float, float, float]
    euler_deg: tuple[float, float, float]


class RocketAsset:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.prim_path = str(config.get("prim_path", "/World/Rocket"))
        self.rigid_body_path = str(config.get("rigid_body_path", self.prim_path))
        self.mjcf_path = resolve_project_path(config.get("mjcf_path", "assets/rocket/rocket.xml"))
        self.usd_path = resolve_project_path(config.get("usd_path", "assets/rocket/rocket.usd"))
        self.mass_kg = float(config.get("mass_kg", 1200.0))
        self.visual_scale = float(config.get("visual_scale", 1.0))
        self.thrust_config = config.get("thrust", {})
        self._rigid_prim: Any | None = None
        self._prim: Any | None = None
        self._converted_usd_path: Path | None = None

    def prepare_usd(self) -> Path:
        if self._converted_usd_path and self._converted_usd_path.exists():
            return self._converted_usd_path
        if self.usd_path.exists() and self.usd_path.stat().st_size > 0:
            print(f"[LunarRocket] using cached rocket USD: {self.usd_path}", flush=True)
            return self.usd_path
        if not self.mjcf_path.exists():
            raise FileNotFoundError(
                "Rocket USD is missing and no MuJoCo model was found. "
                f"Place the MJCF/XML model at {self.mjcf_path} or preconvert it to {self.usd_path}."
            )
        self.usd_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[LunarRocket] converting MJCF to USD: {self.mjcf_path} -> {self.usd_path}", flush=True)
        converted_path = self._convert_mjcf_to_usd()
        if not converted_path.exists():
            raise RuntimeError(f"MJCF conversion completed without creating {converted_path}")
        self._converted_usd_path = converted_path
        return converted_path

    def spawn(self, stage: Any, world: Any | None = None) -> Any:
        usd_path = self.prepare_usd()
        try:
            from pxr import Gf, Sdf, UsdPhysics
            try:
                from isaacsim.core.utils.stage import add_reference_to_stage
            except ImportError:
                from omni.isaac.core.utils.stage import add_reference_to_stage
        except ImportError as exc:
            raise RuntimeError("Isaac Sim Python modules are required to spawn the rocket") from exc

        if stage.GetPrimAtPath(self.prim_path):
            stage.RemovePrim(self.prim_path)

        add_reference_to_stage(str(usd_path), self.prim_path)
        print(f"[LunarRocket] rocket referenced at {self.prim_path}", flush=True)
        self._prim = stage.GetPrimAtPath(self.prim_path)
        if not self._prim:
            raise RuntimeError(f"Rocket prim was not created at {self.prim_path}")

        physics_prim = stage.GetPrimAtPath(self.rigid_body_path) or self._prim
        UsdPhysics.RigidBodyAPI.Apply(physics_prim)
        mass_api = UsdPhysics.MassAPI.Apply(physics_prim)
        mass_api.CreateMassAttr(self.mass_kg)
        physics_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(False)

        return physics_prim

    def reset_pose(
        self,
        position: tuple[float, float, float],
        euler_deg: tuple[float, float, float],
        log: bool = True,
    ) -> RocketState:
        if self._rigid_prim is not None:
            quat = self._euler_xyz_to_quat_wxyz(euler_deg)
            self._rigid_prim.set_world_pose(position=np.array(position), orientation=np.array(quat))
            self._rigid_prim.set_linear_velocity(np.zeros(3))
            self._rigid_prim.set_angular_velocity(np.zeros(3))
        elif self._prim is not None:
            self._set_xform_pose(self._prim, position, euler_deg)
        if log:
            print(
                "[LunarRocket] rocket pose "
                f"position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
                f"euler_deg=({euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}) "
                f"visual_scale={self.visual_scale:.2f}",
                flush=True,
            )
        return RocketState(self.prim_path, position, euler_deg)

    def apply_thrust(
        self,
        throttle: float,
        direction: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        throttle = float(np.clip(throttle, 0.0, 1.0))
        max_thrust = float(self.thrust_config.get("max_newtons", 18000.0))
        vector = np.array(direction or self.thrust_config.get("default_direction", [0.0, 0.0, 1.0]), dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Thrust direction must be non-zero")
        force = vector / norm * max_thrust * throttle

        if self._rigid_prim is not None:
            try:
                self._rigid_prim.apply_force(force)
            except AttributeError:
                pass
        return force

    def _convert_mjcf_to_usd(self) -> Path:
        try:
            import omni.kit.commands
            import omni.kit.app
        except ImportError as exc:
            raise RuntimeError("MJCF conversion must run inside Isaac Sim Python") from exc

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ("isaacsim.asset.importer.mjcf", "omni.importer.mjcf"):
            try:
                manager.set_extension_enabled_immediate(extension, True)
                break
            except Exception:
                continue

        try:
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            config = MJCFImporterConfig(
                mjcf_path=str(self.mjcf_path),
                usd_path=str(self.usd_path.parent),
                import_scene=False,
                merge_mesh=False,
                collision_from_visuals=False,
                allow_self_collision=False,
                fix_base=False,
                run_asset_transformer=False,
                run_multi_physics_conversion=True,
            )
            output_path = Path(MJCFImporter(config).import_mjcf())
            if output_path != self.usd_path:
                shutil.copyfile(output_path, self.usd_path)
                return self.usd_path
            return output_path
        except ImportError:
            pass

        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not status:
            raise RuntimeError("Failed to create MJCF import config")

        for attr, value in {
            "set_fix_base": False,
            "set_import_sites": True,
            "set_make_default_prim": True,
            "set_create_physics_scene": False,
        }.items():
            if hasattr(import_config, attr):
                getattr(import_config, attr)(value)

        command_variants = [
            {
                "mjcf_path": str(self.mjcf_path),
                "import_config": import_config,
                "dest_path": str(self.usd_path),
            },
            {
                "mjcf_path": str(self.mjcf_path),
                "import_config": import_config,
                "usd_path": str(self.usd_path),
            },
        ]
        last_error: Exception | None = None
        for kwargs in command_variants:
            try:
                status, _ = omni.kit.commands.execute("MJCFCreateAsset", **kwargs)
                if status:
                    return self.usd_path
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to convert MJCF to USD: {last_error}")

    def _set_xform_pose(
        self,
        prim: Any,
        position: tuple[float, float, float],
        euler_deg: tuple[float, float, float],
    ) -> None:
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xformable(prim)
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            translate_op = UsdGeom.XformOp(translate_attr)
        else:
            translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat)

        rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr and rotate_attr.IsValid():
            rotate_op = UsdGeom.XformOp(rotate_attr)
        else:
            rotate_op = xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionFloat)

        scale_attr = prim.GetAttribute("xformOp:scale")
        if scale_attr and scale_attr.IsValid():
            scale_op = UsdGeom.XformOp(scale_attr)
        else:
            scale_op = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)

        translate_op.Set(Gf.Vec3f(*position))
        rotate_op.Set(Gf.Vec3f(*euler_deg))
        scale_op.Set(Gf.Vec3f(self.visual_scale, self.visual_scale, self.visual_scale))
        xform.SetXformOpOrder([translate_op, rotate_op, scale_op])

    def _euler_xyz_to_quat_wxyz(self, euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
        roll, pitch, yaw = np.deg2rad(np.array(euler_deg, dtype=float))
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        return (
            float(cr * cp * cy + sr * sp * sy),
            float(sr * cp * cy - cr * sp * sy),
            float(cr * sp * cy + sr * cp * sy),
            float(cr * cp * sy - sr * sp * cy),
        )
