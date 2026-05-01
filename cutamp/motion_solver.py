# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo."""

import logging
from typing import List

import torch
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Sphere
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_domain import MoveHolding, Push, PushStick, MoveFree, Place, Pick
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    timeline: str = "curobo",
):
    """
    Solve for full motion plan given a plan skeleton and optimized particles.
    Note that visualization adds non-trivial overhead.
    """
    plan_skeleton = plan_info["plan_skeleton"]
    motion_gen = world.get_motion_gen(collision_activation_distance=config.world_activation_distance)
    if config.warmup_motion_gen:
        with timer.time("curobo_motion_gen_warmup", log_callback=_log.debug):
            motion_gen.warmup()

    plan_config = MotionGenPlanConfig(
        timeout=0.5, enable_finetune_trajopt=False, time_dilation_factor=config.time_dilation_factor
    )

    # Log initial state
    ts = 0.0
    obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}
    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    last_js = JointState.from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"

    # Top-Down Hover Distance (20cm directly above objects in World Space)
    hover_z_distance = 0.25 

    # Accumulated plans that the real robot can actually execute
    last_op_type = None
    accum_plans = []

    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name

        def is_target_of_action_with_approach(target_q_name):
            for op in plan_skeleton:
                if op.operator.name in ["Pick", "Place", "Detect"] and op.values[-1] == target_q_name:
                    return True
            return False

        # MoveFree
        if op_name == MoveFree.name:
            q_start, traj, q_end = ground_op.values
            if q_end in best_particle:

                # DELEGATION: If moving to a grasp, let Pick handle it for safe approach
                if is_target_of_action_with_approach(q_end):
                    last_q_name = q_start
                    print(f"[{op_name}] Deferring global motion to target {q_end} to the Pick/Place block.")
                    continue

                with timer.time("curobo_planning"):
                    start_js = last_js
                    target_q = best_particle[q_end].clone()
                    target_js = JointState.from_position(target_q[None])

                    # GRASP VERIFICATION PROBE 
                    print(f"\n[GRASP PROBE] Analyzing Target State for MoveFree to {q_end}")
                    try:
                        fk_result = motion_gen.kinematics.compute_kinematics(target_js)
                        
                        if hasattr(fk_result, 'ee_position'):
                            ee_pos = fk_result.ee_position[0].cpu().numpy()
                            ee_quat = fk_result.ee_quaternion[0].cpu().numpy()
                        elif hasattr(fk_result, 'ee_pose'):
                            ee_pos = fk_result.ee_pose.position[0].cpu().numpy()
                            ee_quat = fk_result.ee_pose.quaternion[0].cpu().numpy()
                        else:
                            raise AttributeError("Could not find position attribute.")
                        
                        print(f" -> EE Target XYZ : [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
                        print(f" -> EE Target Quat: [{ee_quat[0]:.4f}, {ee_quat[1]:.4f}, {ee_quat[2]:.4f}, {ee_quat[3]:.4f}]")
                        
                        if ee_pos[2] < 0.43: 
                            print(f" DANGER: End Effector (Z={ee_pos[2]:.4f}) is colliding with or below the table surface (Z=0.425)!")
                    except Exception as e:
                        print(f" Could not compute FK for probe: {e}")

                    result = motion_gen.plan_single_js(start_js, target_js, plan_config)
                    
                    if not result.success:
                        _log.error(f"Failed to plan MoveFree to {q_end}. Status: {result.status}")
                        raise RuntimeError(f"Failed to plan motion for {ground_op.name}")

                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                
                last_q_name = q_end
                last_op_type = "MoveFree"
                
            else:
                last_q_name = q_start

        # MoveHolding
        elif op_name == MoveHolding.name:
            obj, grasp, q_start, traj, q_end = ground_op.values
            if q_end in best_particle:
                
                # DELEGATION: If moving to place, let Place handle it for a safe top-down approach
                if is_target_of_action_with_approach(q_end):
                    last_q_name = q_start
                    print(f"[{op_name}] Deferring global motion to target {q_end} to the Pick/Place block.")
                    continue 
                
                with timer.time("curobo_planning"):
                    start_js = last_js
                    target_q = best_particle[q_end].clone()
                    target_js = JointState.from_position(target_q[None])
                    
                    result = motion_gen.plan_single_js(start_js, target_js, plan_config)
                    if not result.success:
                        raise RuntimeError(f"Failed to plan motion for {ground_op.name}")

                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                last_q_name = q_end
                last_op_type = "MoveHolding"
            else:
                last_q_name = q_start

        # Pick
        elif op_name == Pick.name:
            obj, grasp, q = ground_op.values
            assert last_js is not None

            with timer.time("curobo_planning"):
                start_js = last_js

                target_q = best_particle[q].clone()
                target_js = JointState.from_position(target_q[None])

                # Calculate strictly top-down approach pose (20cm above the PyTorch optimized grasp)
                world_from_ee = world.kin_model.get_state(target_js.position).ee_pose.get_matrix()[0]
                world_from_hover = world_from_ee.clone()
                world_from_hover[2, 3] += hover_z_distance

                # Plan Neutral Retract (Untwisting the wrist before transit)
                world_from_ee_start = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                
                # Create a waypoint 20cm straight up from the current position
                world_from_start_retract = world_from_ee_start.clone()
                world_from_start_retract[2, 3] += hover_z_distance
                
                # FORCE WRIST TO NEUTRAL: Point straight down [Roll=180, Pitch=0, Yaw=0]
                # cuRobo expects top-down grasps to face the table (-Z)
                neutral_rot = torch.tensor([
                    [ 1.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0, -1.0]
                ], dtype=torch.float32, device=world.device)
                world_from_start_retract[:3, :3] = neutral_rot

                retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_start_retract), plan_config)
                
                if retract_result.success:
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    print("WARNING: Could not untwist wrist. Falling back to start state.")
                    retract_result = None
                    retract_js = start_js

                # Plan Global Transit (across the workspace to hover exactly above the object)
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_hover), plan_config)
                if not approach_result.success:
                    raise RuntimeError(f"Failed to plan approach for {ground_op.name}. Status: {approach_result.status}")
                
                # Plan Final Descent (Straight down into the PyTorch grasp)
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                
                # Ghost the object so cuRobo doesn't panic during the descent
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=obj) 
                
                end_result = motion_gen.plan_single_js(approach_js, target_js, plan_config)
                
                if not end_result.success:
                    motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                    raise RuntimeError(f"Failed to plan final grasp insertion for {ground_op.name}. Status: {end_result.status}")

            for result in [retract_result, approach_result, end_result]:
                if result is None:
                    continue
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

            # Temporarily monkey patch get_bounding_spheres to return the spheres we sampled
            obstacle = motion_gen.world_model.get_obstacle(obj)
            obstacle.old_get_bounding_spheres = obstacle.get_bounding_spheres

            def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
                spheres = world.get_collision_spheres(obj)
                pts = spheres[:, :3].cpu().numpy()
                n_radius = spheres[:, 3].cpu().numpy()
                obj_pose = Pose.from_list(self.pose, self.tensor_args)
                pre_transform_pose = kwargs["pre_transform_pose"]
                if pre_transform_pose is not None:
                    obj_pose = pre_transform_pose.multiply(obj_pose)
                points_cuda = self.tensor_args.to_device(pts)
                pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()

                return [
                    Sphere(name=f"{self.name}_sph_{i}", pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0], radius=n_radius[i])
                    for i in range(pts.shape[0])
                ]

            obstacle.get_bounding_spheres = get_bounding_spheres.__get__(obstacle)

            # Attach the object to the robot
            with timer.time("curobo_planning"):
                motion_gen.attach_objects_to_robot(
                    last_js,
                    object_names=[obj],
                    surface_sphere_radius=0.005,
                    sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                    voxelize_method="subdivide",
                )

            obstacle.get_bounding_spheres = obstacle.old_get_bounding_spheres
            del obstacle.old_get_bounding_spheres

            # Close the gripper
            if config.robot == "ur5":
                interp = torch.linspace(0.0, 0.4, 20)[:, None]
            else:
                interp = torch.linspace(0.04, 0.02, 20)[:, None].repeat(1, 2)
            
            accum_plans.append({"type": "gripper", "action": "close"})
            all_pos = torch.cat([last_js.position.expand(interp.shape[0], -1).cpu(), interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=0.02)

            # Plan Lift(Pull object 20cm straight up out of the clutter)
            lift_result = motion_gen.plan_single(last_js, Pose.from_matrix(world_from_hover), plan_config)
            if not lift_result.success:
                raise RuntimeError(f"Failed to plan lift after grasping {ground_op.name}. Status: {lift_result.status}")
            
            dt = lift_result.interpolation_dt
            plan = lift_result.get_interpolated_plan()
            accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
            last_js = JointState.from_position(plan[-1:].position)
            ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
            last_op_type = "Pick"

        # Place
        elif op_name == Place.name:
            obj, grasp, placement, surface, q = ground_op.values
            assert last_js is not None

            with timer.time("curobo_planning"):
                start_js = last_js # The robot is currently hovering safely above the pick site
                
                # Use optimized joint state for the placement
                target_q = best_particle[q].clone()
                target_js = JointState.from_position(target_q[None])

                # Calculate strictly top-down hover pose over the discard zone
                world_from_ee = world.kin_model.get_state(target_js.position).ee_pose.get_matrix()[0]
                world_from_ee_start = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                
                world_from_hover = world_from_ee.clone()
                world_from_hover[2, 3] += hover_z_distance

                # Plan safe crossing above the table to the discard zone
                approach_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_hover), plan_config)
                if not approach_result.success:
                    raise RuntimeError(f"Failed to plan approach for {ground_op.name}. Status: {approach_result.status}")

                # Turn the object into GHOST for the final descent
                # cuRobo will reject the final 2mm if the attached object 
                # touches the discard zone geometry. We temporarily disable
                # the object to let it slide into place.
                motion_gen.detach_object_from_robot("attached_object")
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=obj)

                # Plan Final Descent (Straight down into the PyTorch placement)
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single_js(approach_js, target_js, plan_config)
                if not end_result.success:
                    raise RuntimeError(f"Failed to plan final placement insertion for {ground_op.name}. Status: {end_result.status}")

            # Compute the offset between the object and end-effector while grasped
            obj_from_ee = torch.inverse(obj_to_current_pose[obj]) @ world_from_ee_start
            ee_from_obj = torch.inverse(obj_from_ee)

            for result in [approach_result, end_result]:
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                last_js = JointState.from_position(plan[-1:].position)

                # Forward kinematics to update object position during transit
                robot_state = world.kin_model.get_state(plan.position)
                world_from_ee_traj = robot_state.ee_pose.get_matrix()
                world_from_obj = world_from_ee_traj @ ee_from_obj
                ts = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=plan.position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj,
                    timeline=timeline,
                    start_time=ts,
                    dt=dt,
                )
                obj_to_current_pose[obj] = world_from_obj[-1]

            # Detach object from robot and enable collision again
            with timer.time("curobo_planning"):
                motion_gen.detach_object_from_robot("attached_object")
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                obj_pose = obj_to_current_pose[obj]
                motion_gen.world_collision.update_obstacle_pose(
                    obj, Pose.from_matrix(obj_pose), update_cpu_reference=True
                )

            # Open the gripper
            if config.robot == "ur5":
                interp = torch.linspace(0.4, 0.0, 20)[:, None]
            else:
                interp = torch.linspace(0.02, 0.04, 20)[:, None].repeat(1, 2)
            
            accum_plans.append({"type": "gripper", "action": "open"})
            all_pos = torch.cat([last_js.position.expand(interp.shape[0], -1).cpu(), interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=0.02)

            # Plan Lift(Pull empty gripper 20cm straight up from discard)
            lift_result = motion_gen.plan_single(last_js, Pose.from_matrix(world_from_hover), plan_config)
            if not lift_result.success:
                raise RuntimeError(f"Failed to plan lift after placing {ground_op.name}. Status: {lift_result.status}")
            
            dt = lift_result.interpolation_dt
            plan = lift_result.get_interpolated_plan()
            accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
            last_js = JointState.from_position(plan[-1:].position)
            ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
            last_op_type = "Place"

        elif op_name == Push.name or op_name == PushStick.name:
            raise NotImplementedError("Push and PushStick operations are not yet supported in cuRobo motion planning.")
        
        # Detect
        elif op_name == "Detect":
            obj, pose_name, q_name = ground_op.values
            assert last_js is not None

            with timer.time("curobo_planning"):
                start_js = last_js
                target_q = best_particle[q_name].clone()
                target_js = JointState.from_position(target_q[None])

                # Get the Cartesian pose of the camera target
                world_from_detect_target = world.kin_model.get_state(target_js.position).ee_pose.get_matrix()[0]

                # Safe Hover
                # Prevent arm from sweeping across the cluttered table.
                SAFE_Z_ALTITUDE = 0.45 

                # RETRACT: Go straight up from current position
                world_from_ee_start = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                world_from_start_retract = world_from_ee_start.clone()
                world_from_start_retract[2, 3] = max(world_from_start_retract[2, 3] + 0.1, SAFE_Z_ALTITUDE)
                
                retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_start_retract), plan_config)
                
                if retract_result.success:
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    print("WARNING: Could not plan safe retract. Falling back to start state.")
                    retract_js = start_js

                # Move horizontally above the target
                world_from_detect_hover = world_from_detect_target.clone()
                world_from_detect_hover[2, 3] = max(world_from_detect_target[2, 3] + 0.1, SAFE_Z_ALTITUDE)
                
                transit_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_detect_hover), plan_config)
                
                if transit_result.success:
                    transit_js = JointState.from_position(transit_result.get_interpolated_plan().position[-1:])
                else:
                    print("WARNING: High transit failed. Attempting direct transit.")
                    transit_js = retract_js

                # FINAL DESCENT: Plunge down to the actual Detect pose
                end_result = motion_gen.plan_single_js(transit_js, target_js, plan_config)
                
                # Append the successful trajectory segments
                if retract_result and retract_result.success:
                    dt = retract_result.interpolation_dt
                    plan = retract_result.get_interpolated_plan()
                    accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                    
                if transit_result.success and transit_result is not retract_result:
                    dt = transit_result.interpolation_dt
                    plan = transit_result.get_interpolated_plan()
                    accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                    
                if not end_result.success:
                    raise RuntimeError(f"Failed to plan Detect sequence for {ground_op.name}.")
                
                dt = end_result.interpolation_dt
                plan = end_result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

            winning_pose = best_particle[pose_name].cpu().numpy()
            print("\n" + "="*40)
            print(f"Executing Detect -> Camera shutter triggered at (xyz): {winning_pose[:3]}")
            print("="*40 + "\n")
            
            # Emit the Detect action so the PyBullet executor knows to pause for the callback!
            accum_plans.append({"type": "detect", "target": obj})
            last_op_type = "Detect"
        else:
            raise NotImplementedError(f"Unsupported operator {op_name}")

        print(f"{idx + 1}. {ground_op.name}")

    start_js = last_js

    # Plan to go home at the end
    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
    world_from_retract = world_from_ee.clone()
    world_from_retract[2, 3] += hover_z_distance
    
    retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), plan_config)
    if retract_result.success:
        dt = retract_result.interpolation_dt
        plan = retract_result.get_interpolated_plan()
        accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
        last_js = JointState.from_position(plan[-1:].position)
        ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

    q_last = last_js.position[0]
    q_home = best_particle["q0"].clone()
    js_last = JointState.from_position(q_last[None])
    js_home = JointState.from_position(q_home[None])
    
    with timer.time("curobo_planning"):
        result = motion_gen.plan_single_js(js_last, js_home, plan_config)
    if not result.success:
        print("WARNING: Failed to plan for going home, but returning successful Pick plan!")

    if result.success:
        dt = result.interpolation_dt
        plan = result.get_interpolated_plan()
        accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
        _ = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
        _log.debug("Planned to go home")

    _log.info(f"Motion planning metrics: {timer.get_summary('curobo_planning')}")
    return accum_plans, winning_pose