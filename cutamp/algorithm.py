# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import logging
from datetime import datetime
from typing import Any, Iterator, List, Optional, Tuple, Union
from unittest.mock import Mock

import numpy as np
import pybullet as p
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from imagine_tamp.tamp.belief import BeliefManager
from imagine_tamp.tamp.cutamp_utils import cuTAMPUtilities
from tqdm import tqdm

from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment
from cutamp.experiment_logger import ExperimentLogger
from cutamp.motion_solver import solve_curobo
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import ParticleInitializer
from cutamp.robots import get_q_home, load_robot_container
from cutamp.rollout import RolloutFunction
from cutamp.tamp_domain import all_tamp_operators
from cutamp.tamp_world import TAMPWorld, check_tamp_world_not_in_collision
from cutamp.task_planning import PlanSkeleton, get_top_k_plan_skeletons
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import MockVisualizer, RerunVisualizer

_log = logging.getLogger(__name__)


def heuristic_fn(
    plan_skeleton: PlanSkeleton,
    cost_dict: dict,
    constraint_checker: ConstraintChecker,
    verbose: bool = True,
) -> float:
    """
    Get a single heuristic value for a cost dict corresponding to a rollout.

    We first compute the success rate of each constraint. If the constraint has zero success, we assign it a penalty
    of -num_particles. We then compute the mean success rate across all constraints, and use the failure rate as the
    heuristic (lower the better).
    """
    full_mask = constraint_checker.get_full_mask(cost_dict)
    successes = []
    num_particles = None
    for con_type, con_info in full_mask.items():
        for name, mask in con_info.items():
            if mask.ndim == 2:
                satisfying = mask.sum(0)
            else:
                satisfying = mask.sum()

            if num_particles is None:
                num_particles = mask.shape[0]
            else:
                assert num_particles == mask.shape[0]

            # replace zeros with -num_particles
            satisfying[satisfying == 0] = -num_particles
            successes.extend(satisfying.tolist())
            if verbose:
                _log.debug(f"{con_type} {name} {satisfying.tolist()}")
    success_mean = sum(successes) / len(successes)
    success_rate = success_mean / num_particles
    failure_rate = 1 - success_rate
    heuristic = 100 * failure_rate

    # We have a preference for shorter plans
    heuristic += len(plan_skeleton)
    return heuristic


def get_best_particle(
    plan_info: dict,
    config: TAMPConfiguration,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
) -> dict:
    """Get the particle that satisfies the constraints and has the best soft cost."""
    particles, rollout_fn, cost_fn = (
        plan_info["particles"],
        plan_info["rollout_fn"],
        plan_info["cost_fn"],
    )
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    # Take the best particle that is satisfying and has the best soft cost
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")

    soft_costs = cost_reducer.soft_costs(cost_dict)
    satisfying_costs = soft_costs[satisfying_mask]
    best_satisfying_idx = satisfying_costs.argmin()
    indices = torch.arange(config.num_particles, device=satisfying_costs.device)
    best_idx = indices[satisfying_mask][best_satisfying_idx]
    best_particle = {k: v[best_idx].detach().clone() for k, v in particles.items()}
    return best_particle


def sample_optimistic_grasps(obj_name: str, current_ee_xyz: list, world: TAMPWorld, num_samples=20) -> tuple:
    """Samples top-down grasps, validates them using PyBullet's built-in IK, and returns the closest one."""
    try:
        obj_pose = world.get_object(obj_name).pose
        if torch.is_tensor(obj_pose):
            obj_pose = obj_pose.cpu().numpy()
        obj_xyz = obj_pose[:3]
    except Exception:
        return current_ee_xyz, 0.0, None  # Fallback

    valid_grasps = []

    try:
        # Panda robot in the PyBullet scene
        robot_id = 0
        for i in range(p.getNumBodies()):
            info = p.getBodyInfo(i)
            if b"panda" in info[1].lower() or b"franka" in info[1].lower():
                robot_id = i
                break

        for _ in range(num_samples):
            yaw = np.random.uniform(-np.pi, np.pi)
            for _ in range(num_samples):
                yaw = np.random.uniform(-np.pi, np.pi)

                if obj_xyz[2] < 0.5:
                    # TABLETOP: Top-Down Grasp
                    target_pos = [obj_xyz[0], obj_xyz[1], obj_xyz[2] + 0.13]
                    target_quat = p.getQuaternionFromEuler([np.pi, 0.0, yaw])
                else:
                    # SHELF: Lateral Grasp (Approach from +X towards -X)
                    # Note: Panda lateral Euler is typically Pitch = pi/2
                    target_pos = [obj_xyz[0] + 0.15, obj_xyz[1], obj_xyz[2]]
                    target_quat = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0.0])

            # calculateInverseKinematics returns joint angles. If it executes
            # successfully, the pose is kinematically feasible for heuristic
            p.calculateInverseKinematics(
                bodyUniqueId=robot_id,
                endEffectorLinkIndex=8,  # Panda wrist
                targetPosition=target_pos,
                targetOrientation=target_quat,
                maxNumIterations=20,
                residualThreshold=1e-3,
            )
            valid_grasps.append((target_pos, yaw))

    except Exception as e:
        print(f"  [PyBullet Warning] IK failed: {e}. Falling back to geometry heuristic.")
        # Fallback to pure geometry if PyBullet is disconnected
        for _ in range(num_samples):
            yaw = np.random.uniform(-np.pi, np.pi)
            target_pos = [obj_xyz[0], obj_xyz[1], obj_xyz[2] + 0.13]
            valid_grasps.append((target_pos, yaw))

    if not valid_grasps:
        dist = torch.linalg.norm(torch.tensor(current_ee_xyz) - torch.tensor(obj_xyz)).item()
        return obj_xyz, dist, None

    # Find the valid grasp closest to the arm's current position
    best_grasp_tensor = None
    min_dist = float("inf")
    best_xyz = None

    for pos, yaw in valid_grasps:
        dist = np.linalg.norm(np.array(current_ee_xyz) - np.array(pos))
        if dist < min_dist:
            min_dist = dist
            best_xyz = pos
            best_grasp_tensor = [pos[0], pos[1], pos[2], yaw]

    return best_xyz, min_dist, best_grasp_tensor


def sample_optimistic_placements(
    obj_name: str,
    surface_name: str,
    current_ee_xyz: list,
    world: TAMPWorld,
    num_samples=10,
) -> tuple:
    """Samples placements on a surface, filters for collisions, and returns the closest valid one."""
    try:
        surface_aabb = world.get_aabb(surface_name)
        if torch.is_tensor(surface_aabb):
            surface_aabb = surface_aabb.cpu().numpy()
    except Exception:
        return current_ee_xyz, 0.0, None

    valid_placements = []

    # Sample placements inside the surface bounds
    for _ in range(num_samples):
        # Sample X, Y inside the AABB, padded to avoid edges
        x = np.random.uniform(surface_aabb[0, 0] + 0.05, surface_aabb[1, 0] - 0.05)
        y = np.random.uniform(surface_aabb[0, 1] + 0.05, surface_aabb[1, 1] - 0.05)
        z = surface_aabb[1, 2] + 0.05  # Surface Z + half object height

        place_xyz = [x, y, z]
        yaw = np.random.uniform(-np.pi, np.pi)
        place_pose = place_xyz + [yaw]

        # Check distance to other objects to avoid placing on top of them
        collision = False
        for other_obj in world.movables:
            if other_obj.name != obj_name:
                # Ensure other_xyz is safely handled whether it's a list or a tensor
                other_xyz = other_obj.pose[:3]
                if torch.is_tensor(other_xyz):
                    other_xyz = other_xyz.cpu().numpy()

                if np.linalg.norm(np.array(place_xyz[:2]) - np.array(other_xyz[:2])) < 0.08:  # 8cm clearance
                    collision = True
                    break

        if not collision:
            valid_placements.append((place_xyz, place_pose))

    if not valid_placements:
        return current_ee_xyz, 5.0, None  # Massive penalty if no valid placements found

    # Find the one closest to the arm's current holding position
    best_place = None
    min_dist = float("inf")

    for xyz, pose in valid_placements:
        dist = torch.linalg.norm(torch.tensor(current_ee_xyz) - torch.tensor(xyz)).item()
        if dist < min_dist:
            min_dist = dist
            best_place = pose
            best_xyz = xyz

    return best_xyz, min_dist, best_place


def yield_optimistic_skeletons(
    top_k_skeletons: List,
    scene_config: dict,
    scene_mapping: dict,
    world: TAMPWorld,  
    mission_target_name: str = "ghost",
    target_visible: bool = False,
    penalize_longer_plans: bool = False,
    verbose: bool = False,
) -> Iterator[Tuple[Any, List, dict]]:

    if verbose:
        print("\n" + "=" * 60)
        print("TASK LEVEL: Optimistic Holistic Evaluation")
        print("=" * 60)

    scored_skeletons = []

    # Heuristic Weights
    WEIGHT_EFFORT = 1.0
    WEIGHT_VISIBILITY = 2.5
    longer_plan_penalty = 5.0

    pbar = tqdm(
        enumerate(top_k_skeletons),
        total=len(top_k_skeletons),
        desc="Evaluating Skeletons",
    )

    for idx, skeleton in pbar:
        plan_str = " -> ".join([op.name if not hasattr(op, "operator") else op.operator.name for op in skeleton])

        candidate_poses_dict = {}
        effort_distance = 0.0
        cost_breakdown = []

        # PHASE 1A: PRE-SCAN PLAN (Find Objects Moved & Detects)
        objects_moved = set()
        num_detects = 0

        for op in skeleton:
            op_base_name = op.operator.name if hasattr(op, "operator") else op.name
            if "Pick" in op_base_name:
                target_obj = op.values[0]
                if target_obj != mission_target_name:
                    objects_moved.add(target_obj)
            elif "Detect" in op_base_name:
                num_detects += 1

        # PHASE 1B: GENERATE MASTER NBV (Locked for entire plan)
        current_ee_xyz = [0.0, 0.0, 0.9]  # Initial Robot Home

        vis_cost, master_candidate_poses = cuTAMPUtilities.sample_NBV_for_cutamp(
            env_metadata=(scene_config, scene_mapping),
            plan_str=plan_str,
            target_obj_name=mission_target_name,  # ALWAYS target ghost
            current_ee_xyz=current_ee_xyz,  # Generate relative to home pose
            ignore_objects=list(objects_moved),  # Remove all tracked occluders
            verbose=False,
            motion_lambda=1.0,
            nbv_lambda=5.0,
            visualize_viewpoints=True,
            pixel_scale=150.0,
        )

        optimistic_NBV_cost = vis_cost

        # Resolve the actual XYZ of the master camera pose
        if master_candidate_poses and len(master_candidate_poses) > 0:
            master_nbv_xyz = master_candidate_poses[0][:3]
        else:
            master_nbv_xyz = current_ee_xyz

        # Initial Occlusion Penalty (If plan only looks, and target is hidden)
        if num_detects == 1 and not target_visible:
            effort_distance += 2000.0
            cost_breakdown.append(f"  + Detect({mission_target_name}) OCCLUSION PENALTY: 2000.000m (Target is hidden!)")

        # PHASE 2: SEQUENTIAL EFFORT EVALUATION
        for op in skeleton:
            op_base_name = op.operator.name if hasattr(op, "operator") else op.name

            if "Detect" in op_base_name:
                target = op.values[0]
                pose_var_name = op.values[1]

                # All Detect actions share the same master NBV coordinates
                if not world.has_object(pose_var_name):
                    candidate_poses_dict[pose_var_name] = master_candidate_poses

                # Calculate travel to this camera pose
                dist = torch.linalg.norm(torch.tensor(current_ee_xyz) - torch.tensor(master_nbv_xyz)).item()
                effort_distance += dist

                # Update current_ee_xyz to the camera location
                current_ee_xyz = master_nbv_xyz
                cost_breakdown.append(f"  + Detect({target} @ locked NBV) Travel: {dist:.3f}m")

            elif "Pick" in op_base_name:
                target_obj = op.values[0]
                grasp_var_name = op.values[1]

                best_xyz, dist, best_grasp_pose = sample_optimistic_grasps(target_obj, current_ee_xyz, world)
                effort_distance += dist

                # Do not update current_ee_xyz to grasp location.
                # Keep it anchored to the last Detect pose.

                if best_grasp_pose is not None and not world.has_object(grasp_var_name):
                    candidate_poses_dict[grasp_var_name] = torch.tensor(
                        [best_grasp_pose], dtype=torch.float32, device=world.device
                    )
                cost_breakdown.append(f"  + Pick({target_obj}) Travel: {dist:.3f}m")

            elif "Place" in op_base_name:
                target_obj = op.values[0]
                placement_var_name = op.values[2]
                surface_name = op.values[3]

                if surface_name == "table":
                    effort_distance += 500.0
                    cost_breakdown.append(f"  + Place({target_obj} on table) PENALTY: 500.000m")
                elif surface_name != "discard_zone":
                    effort_distance += 500.0
                    cost_breakdown.append(f"  + Place({target_obj} on {surface_name}) PENALTY: 500.000m")
                else:
                    best_xyz, dist, best_place_pose = sample_optimistic_placements(
                        target_obj, surface_name, current_ee_xyz, world
                    )
                    effort_distance += dist

                    # Do not update current_ee_xyz to place location.

                    if best_place_pose is not None and not world.has_object(placement_var_name):
                        candidate_poses_dict[placement_var_name] = torch.tensor(
                            [best_place_pose], dtype=torch.float32, device=world.device
                        )
                    cost_breakdown.append(f"  + Place({target_obj}) Travel: {dist:.3f}m")

        # PHASE 3: FINAL SCORING
        action_penalty = len(skeleton) * longer_plan_penalty if penalize_longer_plans else 0
        total_effort_cost = (effort_distance * WEIGHT_EFFORT) + action_penalty
        total_vis_cost = optimistic_NBV_cost * WEIGHT_VISIBILITY
        total_task_cost = total_effort_cost + total_vis_cost

        # VERIFICATION OUTPUT
        if verbose:
            print(f"\n[Skeleton {idx + 1} Verification] {plan_str}")
            for step in cost_breakdown:
                print(step)
            print(f"  = Total Traveled: {effort_distance:.3f}m")
            print(f"  = Objects Ignored: {list(objects_moved)}")
            print(f"  = Final Viz Cost : {total_vis_cost:.3f}")
            print(f"  = Final Task Cost (w/ Vis & Penalties): {total_task_cost:.3f}")

        scored_skeletons.append(
            {
                "skeleton": skeleton,
                "cost": total_task_cost,
                "plan_str": plan_str,
                "candidate_poses_dict": candidate_poses_dict,
            }
        )

    scored_skeletons.sort(key=lambda x: x["cost"])

    for rank, item in enumerate(scored_skeletons):
        yield item["cost"], item["skeleton"], item["candidate_poses_dict"]


def sample_plan_skeleton(
    plan_skeleton,
    candidate_poses_dict: dict,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    plan_count: int,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    particle_initializer: ParticleInitializer,
) -> Tuple[Union[dict, None], bool]:
    """
    Try sampling a specific plan skeleton, then its particles and compute the heuristic.
    Returns the plan_info dict and whether any satisfying particles were found upon initialization.
    """

    plan_str = [op.name for op in plan_skeleton]
    _log.debug(f"[Plan {plan_count + 1}] Evaluating plan {plan_str}")

    # Sample particles
    with timer.time("initialize_particles"):
        plan_particles, sampled_grasps = particle_initializer(plan_skeleton)
        print("Sampled 6DOF Grasps Shape: ", sampled_grasps.shape)
    if plan_particles is None:  # failed subgraph
        return None, False

    # Dynamic NBV Candidate Viewpoints Injection
    if candidate_poses_dict:
        for op in plan_skeleton:
            if op.name.startswith("Detect"):
                _log.info(f"Distributing candidate viewpoints for {op.name}")

                # Extract the variable names for this specific Detect action
                params_str = op.name.split("(")[1].replace(")", "")
                parsed_params = [p.strip() for p in params_str.split(",")]
                pose_var_name = parsed_params[1]
                q_var_name = parsed_params[2]

                # Fetch the targeted hemisphere poses specifically for this detect action
                specific_poses = candidate_poses_dict.get(pose_var_name)

                if specific_poses is not None and len(specific_poses) > 0:
                    # Convert list of N poses to a PyTorch tensor
                    candidate_tensor = torch.tensor(specific_poses, dtype=torch.float32, device=world.device)
                    num_candidates = candidate_tensor.shape[0]

                    # Distribute the N poses evenly
                    repeats = config.num_particles // num_candidates
                    remainder = config.num_particles % num_candidates

                    pose_tensor_batch = torch.cat(
                        [
                            candidate_tensor.repeat_interleave(repeats, dim=0),
                            candidate_tensor[:remainder],
                        ],
                        dim=0,
                    )

                    # Overwrite the Cartesian target memory
                    plan_particles[pose_var_name] = pose_tensor_batch

                    # Solve IK for the batch to give the GPU starting seeds
                    world_from_detect = Pose(
                        position=pose_tensor_batch[:, :3],
                        quaternion=pose_tensor_batch[:, 3:],
                    ).get_matrix()

                    world_from_ee = world_from_detect @ world.tool_from_ee
                    ik_result = world.ik_solver.solve_batch(Pose.from_matrix(world_from_ee), seed_config=None)

                    q_sols = ik_result.solution[:, 0].clone()
                    nan_mask = torch.isnan(q_sols).any(dim=1)
                    # Replace any failed IK NaNs with the safe robot home position
                    q_sols[nan_mask] = world.q_init

                    # Overwrite the joint configuration memory
                    plan_particles[q_var_name] = q_sols

                    _log.info(
                        f"Detect IK Success ({pose_var_name}): {ik_result.success.sum().item()}/{config.num_particles}"
                    )
                else:
                    _log.warning(
                        f"No candidate poses found in dictionary for {pose_var_name}. Falling back to random seed."
                    )

    # Fix all unnormalized Quaternions
    for param_name, tensor in plan_particles.items():
        if tensor.shape[-1] == 7:  # If it is a Pose tensor [X, Y, Z, W, X, Y, Z]
            quats = tensor[..., 3:7]
            norms = torch.linalg.norm(quats, dim=-1, keepdim=True)

            # Find any quaternions with a magnitude of 0.0
            zero_mask = (norms == 0.0).squeeze(-1)

            # Force them to be a perfect Identity quaternion [1, 0, 0, 0]
            if zero_mask.any():
                tensor[zero_mask, 3] = 1.0  # Set W to 1.0
                tensor[zero_mask, 4:7] = 0.0  # Set X, Y, Z to 0.0
                print(f"[Sanitizer] Fixed {zero_mask.sum().item()} unnormalized quaternions in '{param_name}'")

    # Rollout particles and compute costs
    rollout_fn = RolloutFunction(plan_skeleton, world, config)
    cost_fn = CostFunction(plan_skeleton, world, config)
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout)
        print(f"\n[COST TRACKER] Analyzing {config.num_particles} particles for: {' -> '.join(plan_str)}")

        found_explosions = False
        # cost_dict usually has keys like ('Pick(obj_1)', 'collision_world')
        for key, val in cost_dict.items():
            # Handle nested dictionaries or direct tensor mapping
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, torch.Tensor):
                        max_val = sub_val.max().item()
                        min_val = sub_val.min().item()
                        if min_val > 0.5:  # If even the best particle has a high cost, it's a geometry problem
                            print(
                                f"  [!] EXPLOSION in {key} -> {sub_key} | Min Cost: {min_val:.3f} | Max: {max_val:.3f}"
                            )
                            found_explosions = True
            elif isinstance(val, torch.Tensor):
                max_val = val.max().item()
                min_val = val.min().item()
                if min_val > 0.5:
                    print(f"  [!] EXPLOSION in {key} | Min Cost: {min_val:.3f} | Max Cost: {max_val:.3f}")
                    found_explosions = True

        if not found_explosions:
            print("  [✓] All actions look mathematically safe! (Min costs < 0.5)")
        print("=====================================================================\n")
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)

    print("Heuristic Cost for this plan: ", heuristic)
    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict)
    num_satisfying = satisfying_mask.sum().item()
    print("Number of solutions for this plan: ", num_satisfying)

    if config.stick_button_experiment and num_satisfying > 0:
        # Custom logic in stick button for breaking early for sampling baseline
        heuristic -= 100
        print(f"Found satisfying plan: {plan_str} heuristic -= 100")

    # Best cost initially
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()
            best_soft_cost = cost_reducer.soft_costs(cost_dict)[satisfying_mask].min().item()
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")

    plan_info = {
        "idx": plan_count,
        "plan_skeleton": plan_skeleton,
        "particles": plan_particles,
        "rollout_fn": rollout_fn,
        "cost_fn": cost_fn,
        "heuristic": heuristic,
        "num_satisfying": num_satisfying,
        "best_cost": best_cost,
        "best_soft_cost": best_soft_cost,
    }

    _log.debug(
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, heuristic = {plan_info['heuristic']}"
    )

    return plan_info, num_satisfying > 0, sampled_grasps


def resample_plan_info(
    plan_info: dict,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    particle_initializer: ParticleInitializer,
) -> int:
    """
    Sample particles again in-place for a plan info container with a plan skeleton. This can be used for rejection
    sampling strategy (for the sampling baseline), or for random restarts.

    Returns number of satisfying particles after re-sampling.
    """
    with timer.time("initialize_particles"), timer.time("resample_particles"):
        plan_particles = particle_initializer(plan_info["plan_skeleton"], verbose=False)

    # Rollout new particles and compute costs
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = plan_info["rollout_fn"](plan_particles)
        cost_dict = plan_info["cost_fn"](rollout)
        heuristic = heuristic_fn(plan_info["plan_skeleton"], cost_dict, constraint_checker, verbose=False)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    num_satisfying = satisfying_mask.sum().item()

    # Best cost
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()  # note: should consider satisfying mask?
            soft_costs = cost_reducer.soft_costs(cost_dict)
            best_soft_cost = soft_costs[satisfying_mask].min().item()
            indices = torch.arange(config.num_particles, device=soft_costs.device)
            best_idx = indices[satisfying_mask][costs[satisfying_mask].argmin()]
            best_soft_idx = indices[satisfying_mask][soft_costs[satisfying_mask].argmin()]
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")
            best_idx = None
            best_soft_idx = None

    # Update plan info
    plan_info["particles"] = plan_particles
    plan_info["heuristic"] = heuristic
    plan_info["num_satisfying"] = num_satisfying
    plan_info["best_cost"] = best_cost
    plan_info["best_soft_cost"] = best_soft_cost
    plan_info["rollout"] = rollout
    plan_info["best_idx"] = best_idx
    plan_info["best_soft_idx"] = best_soft_idx
    return num_satisfying


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
):
    # Validate args and setup experiment logger
    validate_tamp_config(config)
    if experiment_id is None:
        experiment_id = datetime.now().isoformat().split(".")[0]

    exp_logger = ExperimentLogger(name=experiment_id, config=config) if config.enable_experiment_logging else Mock()
    exp_logger.save_env(env)

    # Loading robot can be done offline, so doesn't count towards timing
    tensor_args = TensorDeviceType()
    robot_container = load_robot_container(config.robot, tensor_args)
    if q_init is None:
        q_init = get_q_home(config.robot)
    q_init = tensor_args.to_device(q_init)

    # Load TAMP world and warmup IK solver
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env,
            tensor_args,
            robot=robot_container,
            q_init=q_init,
            collision_activation_distance=config.world_activation_distance,
            coll_n_spheres=config.coll_n_spheres,
            coll_sphere_radius=config.coll_sphere_radius,
        )
        check_tamp_world_not_in_collision(world)

    if config.warmup_ik:
        with timer.time("warmup_ik_solver", log_callback=_log.info):
            world.warmup_ik_solver(config.num_particles)

    # Setup visualizer (doesn't count towards timing)
    visualizer = (
        RerunVisualizer(
            config,
            q_init,
            application_id=env.name,
            recording_id=experiment_id,
            spawn=config.rr_spawn,
        )
        if config.enable_visualizer
        else MockVisualizer()
    )
    visualizer.log_tamp_world(world)
    return exp_logger, visualizer, timer, world


def run_cutamp(
    env: TAMPEnvironment,
    scene_config: dict,
    scene_mapping: dict,
    global_belief: BeliefManager,
    config: TAMPConfiguration,
    mission_target_name: str,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
    verbose: bool = False,
    num_plan_skeletons: int = 30,
    is_shelf_scene: bool = True,
):
    """Overall cuTAMP algorithm implementation."""

    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id)
    particle_initializer = ParticleInitializer(world, config)

    # Task plan generator
    _log.info(f"Initial State: {world.initial_state}")
    _log.info(f"Goal State: {world.goal_state}")

    # Yield one plan at a time(original cuTAMP)
    # with timer.time("get_plan_generator", log_callback=_log.info):
    #     plan_gen = task_plan_generator(
    #         world.initial_state,
    #         world.goal_state,
    #         operators=all_tamp_operators,
    #         explored_state_check=config.explored_state_check,
    #     )

    # Retrieve top K Plan skeletons
    K = num_plan_skeletons
    with timer.time("get_top_k_plans", log_callback=_log.info):
        top_k_skeletons = get_top_k_plan_skeletons(
            world.initial_state,
            world.goal_state,
            operators=all_tamp_operators,
            k=K,
            explored_state_check=config.explored_state_check,
        )

    # Isolation Test
    # cuTAMPUtilities.test_symbolic_task_planner(top_k_skeletons, verbose=True)

    # Select Best skeleton based on Optimistic NBV simulation
    with timer.time("optimistic_evaluation"):
        optimistic_plan_gen = yield_optimistic_skeletons(
            top_k_skeletons=top_k_skeletons,
            scene_config=scene_config,
            scene_mapping=scene_mapping,
            world=world,
            mission_target_name=mission_target_name,
            target_visible=False,
            penalize_longer_plans=False,
            verbose=verbose,
        )

    # Heuristic Evaluation
    # Sample initial plans and particles
    found_solution_initially = False
    num_skipped_plans = 0
    with timer.time("sample_initial_plans", log_callback=_log.info):
        plan_queue: List[dict] = []
        plan_count = 0
        for idx in range(config.num_initial_plans):
            try:
                custom_task_cost, plan_gen, candidate_poses_dict = next(optimistic_plan_gen)

                # ==================================================
                # --- MANUAL SKELETON TOGGLE FOR DEBUGGING ---
                # ==================================================
                # TEST_DETECT_ONLY = False
                # TEST_PICK_ONLY = False

                # truncated_skeleton = []
                # for op in plan_gen:
                #     truncated_skeleton.append(op)
                #     op_name = op.operator.name if hasattr(op, "operator") else op.name

                #     if TEST_DETECT_ONLY and not TEST_PICK_ONLY and "Detect" in op_name:
                #         break
                #     elif TEST_DETECT_ONLY and TEST_PICK_ONLY and "Pick" in op_name:
                #         break  # Stop immediately after the Pick!
                #     elif not TEST_DETECT_ONLY and not TEST_PICK_ONLY and "Place" in op_name:
                #         break  # Stop immediately after the first Place!

                # plan_gen = truncated_skeleton

                # print("\n" + "=" * 60)
                # mode = "PICK ONLY" if TEST_PICK_ONLY else "PICK AND PLACE"
                # print(f" [DEBUG] EXECUTING ISOLATED SKELETON ({mode} - Length: {len(plan_gen)}):")
                # print(" -> ".join([op.name for op in plan_gen]))
                # print("=" * 60 + "\n")
                # ==================================================

                # Shelf Scene Auto Pruning
                if is_shelf_scene:
                    last_detect_idx = -1

                    for i, op in enumerate(plan_gen):
                        op_name = op.operator.name if hasattr(op, "operator") else op.name

                        if "Detect" in op_name:
                            last_detect_idx = i

                    # Keep everything up to and including the last Detect
                    if last_detect_idx != -1:
                        plan_gen = plan_gen[: last_detect_idx + 1]

                plan_info, has_solution, sampled_grasps = sample_plan_skeleton(
                    plan_gen,
                    candidate_poses_dict,
                    world,
                    config,
                    timer,
                    idx,
                    constraint_checker,
                    cost_reducer,
                    particle_initializer,
                )
                if plan_info is None:
                    _log.debug("failed subgraph, skipping...")
                    num_skipped_plans += 1
                    continue
                plan_info["custom_task_cost"] = custom_task_cost
            except StopIteration:
                _log.info("Ran out of plans to sample")
                break
            plan_queue.append(plan_info)
            if has_solution:
                found_solution_initially = True
                break
            plan_count += 1

    # Sort plans by heuristic
    def sort_plans():
        with timer.time("sort_plans"):
            plan_queue.sort(key=lambda x: (x["custom_task_cost"], x["heuristic"]))

    sort_plans()

    # Extract the absolute best skeleton
    best_plan_info = plan_queue[0]
    best_skeleton = best_plan_info["plan_skeleton"]
    best_plan_str = " -> ".join([op.name for op in best_skeleton])

    print("\n" + "=" * 60)
    print("BEST PLAN SKELETON SELECTED")
    print(f"Plan Sequence:  {best_plan_str}")
    print(f"Total Heuristic Cost: {best_plan_info['heuristic']:.2f}")
    print("=" * 60 + "\n")

    # Pass only the best skeleton to cuTAMP optimizer
    plan_queue = [best_plan_info]

    _log.info(f"Num plans evaluated: {len(top_k_skeletons)}, num skipped: {num_skipped_plans}")

    overall_metrics = {
        "num_optimized_plans": 0,
        "num_initial_plans": len(top_k_skeletons),
        "num_skipped_plans": num_skipped_plans,
        "num_satisfying_final": 0,
        "num_particles": config.num_particles,
        "best_cost": float("inf"),
        "best_soft_cost": float("inf"),
    }

    curobo_plan = None
    winning_pose = None
    found_solution = False
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker)

    timer.start("first_solution")
    if found_solution_initially:
        found_solution = True
        timer.stop("first_solution")

    # Optimization loop for each skeleton and its particles
    timer.start("start_optimization")
    for idx, plan_info in enumerate(plan_queue):
        opt_iter = idx + 1
        should_break = False
        plan_skeleton = plan_info["plan_skeleton"]
        _log.info(
            f"[Opt {opt_iter}] Optimizing plan {[op.name for op in plan_skeleton]}, plan idx = {plan_info['idx']}, heuristic = {plan_info['heuristic']:.2f}"
        )
        best_particle = None

        if config.approach == "optimization":
            has_satisfying, metrics, time_exceeded = particle_optimizer(plan_info, timer, visualizer)
            if metrics["best_cost"] is not None:
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], metrics["best_cost"])
            if metrics["best_soft_cost"] is not None:
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], metrics["best_soft_cost"])
            if time_exceeded:
                _log.info("Max loop duration reached, stopping optimization")
                should_break = True
            exp_logger.log_dict(f"optimization/opt_{opt_iter:04d}", metrics)
            if has_satisfying:
                best_particle = get_best_particle(plan_info, config, constraint_checker, cost_reducer)
        else:
            # This is the parallelized sampling baseline
            assert config.approach == "sampling"
            num_resample_attempts = 0
            resample_dur = 0.0
            has_satisfying = plan_info["num_satisfying"] > 0
            total_num_satisfying = plan_info["num_satisfying"]
            best_particle = None
            best_soft_costs = []
            elapsed = []

            if not has_satisfying or not config.break_on_satisfying:
                timer.start("resample_duration")
                for resample_idx in range(config.num_resampling_attempts):
                    if config.max_loop_dur is not None and timer.elapsed("start_optimization") >= config.max_loop_dur:
                        _log.info("Max loop duration reached, stopping resampling")
                        should_break = True
                        break
                    timer.start("resample_plan_info")
                    num_satisfying = resample_plan_info(
                        plan_info,
                        world,
                        config,
                        timer,
                        cost_reducer,
                        constraint_checker,
                        particle_initializer,
                    )
                    total_num_satisfying += num_satisfying
                    if plan_info["best_soft_cost"] < overall_metrics["best_soft_cost"]:
                        best_soft_idx = plan_info["best_soft_idx"]
                        best_particle = {
                            k: v[best_soft_idx].detach().clone() for k, v in plan_info["particles"].items()
                        }

                    overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                    overall_metrics["best_soft_cost"] = min(
                        overall_metrics["best_soft_cost"], plan_info["best_soft_cost"]
                    )

                    # Keep track of the best soft cost since start of resampling
                    best_soft_costs.append(overall_metrics["best_soft_cost"])
                    elapsed.append(timer.elapsed("start_optimization"))

                    resample_plan_info_dur = timer.stop("resample_plan_info")
                    _log.debug(
                        f"[Plan {plan_info['idx'] + 1}] Resample attempt {resample_idx + 1}/{config.num_resampling_attempts}, {num_satisfying}/{config.num_particles} satisfying particles. Total satisfying {total_num_satisfying}. Took {resample_plan_info_dur:.2f}s"
                    )
                    has_satisfying = num_satisfying > 0
                    num_resample_attempts += 1

                    # Visualize best particle rollout state
                    rollout = plan_info["rollout"]
                    best_soft_idx = plan_info["best_soft_idx"]
                    if best_soft_idx is None:
                        best_soft_idx = 0
                    visualizer.set_time_sequence("samp", num_resample_attempts)
                    q_last = rollout["confs"][best_soft_idx, -1].tolist()
                    visualizer.set_joint_positions(q_last)
                    for obj in rollout["obj_to_pose"]:
                        mat4x4_last = rollout["obj_to_pose"][obj][best_soft_idx, -1]
                        visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

                    if has_satisfying:
                        if timer.has_timer("first_solution"):
                            time_to_first_sol = timer.stop("first_solution")
                            _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")
                        if config.break_on_satisfying:
                            should_break = True
                            break
                resample_dur = timer.stop("resample_duration")
                _log.info(f"Total resample duration: {resample_dur:.2f}s")
            else:
                _log.info("Already has satisfying particles, skipping resampling")
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], plan_info["best_soft_cost"])
                if config.break_on_satisfying:
                    should_break = True

            metrics = {
                "plan_skeleton": [str(op) for op in plan_skeleton],
                "num_particles": config.num_particles,
                "num_resample_attempts": num_resample_attempts,
                "resample_duration": resample_dur,
                "num_satisfying_final": total_num_satisfying,
                "total_num_particles": config.num_particles * (num_resample_attempts + 1),
                "best_cost": overall_metrics["best_cost"],
                "best_soft_cost": overall_metrics["best_soft_cost"],
                "best_soft_costs": best_soft_costs,
                "elapsed": elapsed,
            }
            exp_logger.log_dict(f"sampling/samp_{opt_iter:04d}", metrics)
            has_satisfying = total_num_satisfying > 0
            overall_metrics["num_satisfying_final"] = total_num_satisfying

            # Log best particle as last
            if best_particle is not None:
                rollout = plan_info["rollout_fn"]({k: v[None] for k, v in best_particle.items()})
                visualizer.set_time_sequence("samp", num_resample_attempts)
                q_last = rollout["confs"][0, -1].tolist()
                visualizer.set_joint_positions(q_last)

                for obj in rollout["obj_to_pose"]:
                    mat4x4_last = rollout["obj_to_pose"][obj][0, -1]
                    visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

        # Now we've either optimized or resampled
        overall_metrics["num_optimized_plans"] += 1
        if has_satisfying:
            found_solution = True
            if config.curobo_plan:
                curobo_plan, winning_pose_dict = solve_curobo(
                    plan_info,
                    best_particle,
                    world,
                    config,
                    timer,
                    visualizer,
                    is_shelf_scene=is_shelf_scene,
                )
            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            _log.debug(f"Total num satisfying {metrics['num_satisfying_final']}")
            if config.break_on_satisfying:
                should_break = True

        if should_break:
            break

        # TODO: complete version of our algorithm that adds additional skeletons to the queue, resorts, revisits
        #  skeletons, etc.
        # new_plan_info = sample_plan_skeleton()
        # if new_plan_info is not None:
        #     plan_queue.append(new_plan_info)
        #     sort_plans()

    opt_elapsed = timer.stop("start_optimization")
    _log.debug(f"Optimization loop took roughly {opt_elapsed:.2f}s")
    if not found_solution:
        _log.warning("No satisfying particles found after optimizing all plans")
    _log.debug(f"Best cost: {overall_metrics['best_cost']:.4f}, soft cost: {overall_metrics['best_soft_cost']:.4f}")

    # Dump metrics out
    overall_metrics["found_solution"] = found_solution
    exp_logger.log_dict("overall_metrics", overall_metrics)
    exp_logger.log_dict("timer_metrics", timer.get_summaries())

    # Log constraint and cost multipliers
    exp_logger.log_dict("multipliers", cost_reducer.cost_config)
    exp_logger.log_dict("tolerances", constraint_checker.constraint_config)
    return curobo_plan, winning_pose_dict, sampled_grasps, overall_metrics["num_satisfying_final"]


"""
# TODO:

i) Should I add AnyGrasp instead of sampling uniform grasps? For the final grasp pose, not the optimistic grasp pose.
ii) If cuTAMP does not work properly, should I just shift to PB based motion generation? 

"""
