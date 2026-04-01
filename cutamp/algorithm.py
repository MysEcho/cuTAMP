# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import logging
from datetime import datetime
from typing import List, Union, Optional, Tuple, Any, Iterator
from unittest.mock import Mock

import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

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
from cutamp.task_planning import PlanSkeleton, task_plan_generator, get_top_k_plan_skeletons
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, MockVisualizer
from imagine_tamp.tamp.cutamp_utils import cuTAMPUtilities
from imagine_tamp.tamp.belief import BeliefManager


_log = logging.getLogger(__name__)


def heuristic_fn(
    plan_skeleton: PlanSkeleton, cost_dict: dict, constraint_checker: ConstraintChecker, verbose: bool = True
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
    plan_info: dict, config: TAMPConfiguration, constraint_checker: ConstraintChecker, cost_reducer: CostReducer
) -> dict:
    """Get the particle that satisfies the constraints and has the best soft cost."""
    particles, rollout_fn, cost_fn = plan_info["particles"], plan_info["rollout_fn"], plan_info["cost_fn"]
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

def yield_optimistic_skeletons(
    top_k_skeletons: List,
    global_belief:BeliefManager, 
    scene_config: dict,
    scene_mapping: dict,
    penalize_longer_plans:bool=False,
) -> Iterator[Tuple[Any, List]]:
    """
    Evaluates Top K skeletons using optimistic visibility math, sorts them by cost,
    and yields them one by one (best plan first).
    """
    print("\n" + "="*50)
    print("TASK LEVEL: Evaluating and Sorting Top K Skeletons")
    print("="*50)

    # Sample candidate NBV viewpoints once
    print("[Global Eval] Sampling candidate viewpoints for current belief state...")
    
    optimistic_NBV_cost, master_candidate_poses = cuTAMPUtilities.sample_NBV_for_cutamp(
        global_belief=global_belief,
        env_metadata=(scene_config, scene_mapping),
        plan_skeleton=top_k_skeletons[0], 
        plan_str="Global Scene Evaluation"
    )

    scored_skeletons = []
    action_penalty = 0
    longer_plan_penalty = 5

    # Sort the top-k plan skeletons
    for idx, skeleton in enumerate(top_k_skeletons):
        plan_str = " -> ".join([op.name for op in skeleton])
        
        # Penalize longer plans 
        if penalize_longer_plans:
            action_penalty = len(skeleton) * longer_plan_penalty
        
        # Add the global visibility cost
        # We need to make sure that optimistic costs are added based on the actions that are present in the skeleton. For eg.,
        # NBV cost will only be added to plans having Detect action.
        # NOTE: Currently every plan will start with MoveFree -> Detect so we need to apply these costs first, execute the plan, update belief
        # and then execute the rest of the plan with newer optimistic cost and then figure out which skeleton is the best. This has to be done
        # on a receding horizon basis.
        total_task_cost = action_penalty + optimistic_NBV_cost

        print(f"[Task Eval] Skeleton {idx+1} | Length: {action_penalty} | Vis Cost: {optimistic_NBV_cost:.2f} | Total: {total_task_cost:.2f}")
        
        scored_skeletons.append({
            "skeleton": skeleton,
            "cost": total_task_cost,
            "plan_str": plan_str
        })

    # Sort from lowest cost to highest cost
    scored_skeletons.sort(key=lambda x: x["cost"])

    print("="*50)
    print(f"Successfully ranked {len(scored_skeletons)} plans.")
    print("="*50 + "\n")

    for rank, item in enumerate(scored_skeletons):
        print(f"\n[Generator] Yielding Rank {rank+1} Plan (Task Cost: {item['cost']:.2f})")
        print(f"Sequence: {item['plan_str']}")
        
        yield item["skeleton"], master_candidate_poses


def sample_plan_skeleton(
    plan_skeleton,
    candidate_poses_list:List,  
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
        plan_particles = particle_initializer(plan_skeleton)
    if plan_particles is None:  # failed subgraph
        return None, False
    
    # Dynamic NBV Candidate Viewpoints Injection
    if candidate_poses_list is not None:
        for op in plan_skeleton:
            if op.name.startswith("Detect"):
                _log.info("Distributing N candidate viewpoints across particles")
                
                params_str = op.name.split("(")[1].replace(")", "")
                parsed_params = [p.strip() for p in params_str.split(",")]
                pose_var_name = parsed_params[1] 
                q_var_name = parsed_params[2]
                
                # list of N poses to a PyTorch tensor
                candidate_tensor = torch.tensor(candidate_poses_list, dtype=torch.float32, device=world.device)
                num_candidates = candidate_tensor.shape[0]
                
                # Distribute the N poses evenly across the 1024 particles
                repeats = config.num_particles // num_candidates
                remainder = config.num_particles % num_candidates
                
                pose_tensor_batch = torch.cat([
                    candidate_tensor.repeat_interleave(repeats, dim=0),
                    candidate_tensor[:remainder]
                ], dim=0)
                
                # Overwrite the Cartesian target memory
                plan_particles[pose_var_name] = pose_tensor_batch
                
                # Solve IK for the batch to give the GPU starting seeds
                world_from_detect = Pose(
                    position=pose_tensor_batch[:, :3], 
                    quaternion=pose_tensor_batch[:, 3:]
                ).get_matrix()
                
                world_from_ee = world_from_detect @ world.tool_from_ee
                ik_result = world.ik_solver.solve_batch(Pose.from_matrix(world_from_ee), seed_config=None)
                
                # Overwrite the joint configuration memory
                plan_particles[q_var_name] = ik_result.solution[:, 0]
                
                _log.info(f"Detect IK Success: {ik_result.success.sum().item()}/{config.num_particles}")
                break

    # Rollout particles and compute costs
    rollout_fn = RolloutFunction(plan_skeleton, world, config)
    cost_fn = CostFunction(plan_skeleton, world, config)
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout)
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict)
    num_satisfying = satisfying_mask.sum().item()

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
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, "
        f"heuristic = {plan_info['heuristic']}"
    )
    return plan_info, num_satisfying > 0

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
        RerunVisualizer(config, q_init, application_id=env.name, recording_id=experiment_id, spawn=config.rr_spawn)
        if config.enable_visualizer
        else MockVisualizer()
    )
    visualizer.log_tamp_world(world)
    return exp_logger, visualizer, timer, world


def run_cutamp(
    env: TAMPEnvironment,
    scene_config:dict, 
    scene_mapping:dict,
    global_belief:BeliefManager,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
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
    K = 30
    with timer.time("get_top_k_plans", log_callback=_log.info):
        top_k_skeletons = get_top_k_plan_skeletons(
            world.initial_state,
            world.goal_state,
            operators=all_tamp_operators,
            k=K,
            explored_state_check=config.explored_state_check,
        )

    # Isolation Test
    # cuTAMPUtilities.test_symbolic_task_planner(top_k_skeletons)

    # Select Best skeleton based on Optimistic NBV simulation
    with timer.time("optimistic_evaluation"):
        optimistic_plan_gen= yield_optimistic_skeletons(
            top_k_skeletons, global_belief, scene_config, scene_mapping, penalize_longer_plans=True
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
                plan_gen, candidate_poses_list = next(optimistic_plan_gen)
                plan_info, has_solution = sample_plan_skeleton(
                    plan_gen, candidate_poses_list, world, config, timer, idx, constraint_checker, cost_reducer, particle_initializer
                )
                if plan_info is None:
                    _log.debug("failed subgraph, skipping...")
                    num_skipped_plans += 1
                    continue
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
            plan_queue.sort(key=lambda x: x["heuristic"])

    sort_plans()
    
    # Extract the absolute best skeleton
    best_plan_info = plan_queue[0]
    best_skeleton = best_plan_info["plan_skeleton"]
    best_plan_str = " -> ".join([op.name for op in best_skeleton])
    
    print("\n" + "="*60)
    print("BEST PLAN SKELETON SELECTED")
    print(f"Plan Sequence:  {best_plan_str}")
    print(f"Total Heuristic Cost: {best_plan_info['heuristic']:.2f}")
    print("="*60 + "\n")

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
            f"[Opt {opt_iter}] Optimizing plan {[op.name for op in plan_skeleton]}, plan idx = {plan_info['idx']}, "
            f"heuristic = {plan_info['heuristic']:.2f}"
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
                        f"[Plan {plan_info['idx'] + 1}] Resample attempt {resample_idx + 1}/{config.num_resampling_attempts}, "
                        f"{num_satisfying}/{config.num_particles} satisfying particles. Total satisfying {total_num_satisfying}. "
                        f"Took {resample_plan_info_dur:.2f}s"
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
                curobo_plan, winning_pose = solve_curobo(
                    plan_info,
                    best_particle,
                    world,
                    config,
                    timer,
                    visualizer,
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
    return curobo_plan, winning_pose, overall_metrics["num_satisfying_final"]
