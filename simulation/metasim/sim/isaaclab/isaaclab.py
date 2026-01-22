from __future__ import annotations

from types import SimpleNamespace
import torch
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.utils.state import TensorState, RobotState

# IsaacLab (only used to satisfy imports + start Kit so omni/carb are available)
from omni.isaac.lab.app import AppLauncher

# -----------------------------------------------------------------------------
# OFFLINE / STUB IsaacLab backend
# - NO real physics
# - Goal: let MetaSim + pour_demo/collect_pouring_data run end-to-end
# -----------------------------------------------------------------------------

_APP = None


def _dummy_camera_state(num_envs: int, device: torch.device):
    """
    Return a minimal camera-like object to satisfy pour_demo's add_observation().
    We don't assume metasim.utils.state.CameraState signature; use SimpleNamespace.
    """
    H, W = 128, 128
    rgb = torch.zeros((num_envs, H, W, 3), device=device, dtype=torch.uint8)
    depth = torch.zeros((num_envs, H, W), device=device, dtype=torch.float32)

    # common fields various code may try to read
    return SimpleNamespace(
        rgb=rgb,
        depth=depth,
        intrinsics=torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1),
        extrinsics=torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1),
    )


def _make_fake_tensor_state(handler: "IsaaclabHandler") -> TensorState:
    """
    Create a TensorState that is compatible with metasim.utils.state.state_tensor_to_nested()
    and with pour_demo's access patterns.
    """
    robot_cfg = handler.robot_cfg
    num_envs = handler.num_envs
    device = handler.device

    robot_name = robot_cfg.name
    ee_name = getattr(robot_cfg, "ee_body_name", "ee")

    # ---- body names: at least contain ee ----
    body_names = [ee_name]

    # ---- joints: try to derive names & dof count from cfg ----
    joint_names = list(getattr(robot_cfg, "joint_limits", {}).keys())
    dof = int(getattr(robot_cfg, "num_joints", 0) or 0)

    # joint_limits keys might be more than num_joints -> trim
    if dof > 0 and len(joint_names) >= dof:
        joint_names = joint_names[:dof]
    elif dof == 0:
        dof = len(joint_names)

    # final fallback
    if dof <= 0:
        dof = 9
        joint_names = [f"joint_{i+1}" for i in range(dof)]

    # ---- tensors ----
    # root_state: (pos3, quat4, linvel3, angvel3) => 13
    root_state = torch.zeros((num_envs, 13), device=device, dtype=torch.float32)
    root_state[:, 2] = 0.8  # z

    body_state = torch.zeros((num_envs, len(body_names), 13), device=device, dtype=torch.float32)
    body_state[:, 0, :13] = root_state

    joint_pos = torch.zeros((num_envs, dof), device=device, dtype=torch.float32)
    joint_vel = torch.zeros((num_envs, dof), device=device, dtype=torch.float32)
    joint_pos_target = torch.zeros((num_envs, dof), device=device, dtype=torch.float32)
    joint_vel_target = torch.zeros((num_envs, dof), device=device, dtype=torch.float32)
    joint_effort_target = torch.zeros((num_envs, dof), device=device, dtype=torch.float32)

    robots = {
        robot_name: RobotState(
            root_state=root_state,
            body_names=body_names,
            body_state=body_state,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_pos_target=joint_pos_target,
            joint_vel_target=joint_vel_target,
            joint_effort_target=joint_effort_target,
        )
    }

    # add ONE dummy camera so add_observation() has something to read
    cameras = {"default": _dummy_camera_state(num_envs, device)}

    return TensorState(objects={}, robots=robots, cameras=cameras, sensors={})


class IsaaclabHandler(BaseSimHandler):
    """
    Minimal handler satisfying BaseSimHandler interface used by pour_demo.
    IMPORTANT: BaseSimHandler defines num_envs/device as properties (read-only),
    so we store _num_envs/_device internally and expose properties.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self.scenario = scenario
        self.robot_cfg = scenario.robot

        self._num_envs = int(scenario.num_envs)
        self._device = torch.device("cpu")

        self._t = 0
        self._cached_states: TensorState | None = None

        # some scripts access env.handler.env.sim.step()
        self.env = SimpleNamespace(sim=SimpleNamespace(step=lambda: None))

    # ---- required properties ----
    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------
    def launch(self):
        global _APP
        if _APP is None:
            _APP = AppLauncher(headless=self.scenario.headless).app
            log.info("IsaacLab App created (singleton).")
        log.info("IsaacLab backend launch OK (offline stub).")

    def close(self):
        # do not close global app (avoid repeated create/destroy issues)
        return

    # -------------------------------------------------------------------------
    # core API
    # -------------------------------------------------------------------------
    def reset(self, states=None):
        self._t = 0
        self._cached_states = _make_fake_tensor_state(self)
        obs = {"states": [states] if states is not None else [], "t": self._t}
        return obs, {}

    def step(self, action=None):
        self._t += 1
        obs = {"states": [None], "t": self._t}
        reward = 0.0
        terminated = False
        truncated = False
        info = {"t": self._t}
        return obs, reward, terminated, truncated, info

    def get_states(self, *args, **kwargs) -> TensorState:
        if self._cached_states is None:
            self._cached_states = _make_fake_tensor_state(self)
        return self._cached_states

    # -------------------------------------------------------------------------
    # helpers used by pour_demo/state_tensor_to_nested
    # -------------------------------------------------------------------------
    def get_joint_names(self, obj_name: str, sort: bool = True):
        if obj_name != self.robot_cfg.name:
            return []
        names = list(getattr(self.robot_cfg, "joint_limits", {}).keys())
        dof = int(getattr(self.robot_cfg, "num_joints", 0) or 0)
        if dof > 0 and len(names) >= dof:
            names = names[:dof]
        return names

    def get_body_names(self, obj_name: str):
        if obj_name != self.robot_cfg.name:
            return []
        st = self.get_states()
        return list(st.robots[self.robot_cfg.name].body_names)

    def get_camera_names(self):
        st = self.get_states()
        return list(getattr(st, "cameras", {}).keys())

    def refresh_render(self):
        # offline: nothing to render
        return

    # -------------------------------------------------------------------------
    # offline shortcuts (optional)
    # -------------------------------------------------------------------------
    def _offline_set_ee_pose(self, pos: torch.Tensor, quat: torch.Tensor):
        """
        Directly write EE pose into cached TensorState.
        pos: (N,3), quat: (N,4) (x,y,z,w)
        """
        st = self.get_states()
        r = st.robots[self.robot_cfg.name]
        r.root_state[:, 0:3] = pos.to(r.root_state.device).to(r.root_state.dtype)
        r.root_state[:, 3:7] = quat.to(r.root_state.device).to(r.root_state.dtype)
        # keep body 0 consistent (ee)
        r.body_state[:, 0, 0:3] = r.root_state[:, 0:3]
        r.body_state[:, 0, 3:7] = r.root_state[:, 3:7]


class IsaaclabEnv:
    def __init__(self, scenario: ScenarioCfg):
        self.handler = IsaaclabHandler(scenario)
        self.handler.launch()

    def reset(self, *args, **kwargs):
        return self.handler.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.handler.step(*args, **kwargs)

    def get_states(self, *args, **kwargs):
        return self.handler.get_states(*args, **kwargs)

    def close(self):
        return self.handler.close()
