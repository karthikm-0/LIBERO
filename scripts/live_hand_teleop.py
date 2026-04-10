import logging
import pathlib
import sys
import os

# Add robot-lightning to python path so we can import our HandController
sys.path.append("/home/karthikm/robot-lightning")
from robots.hand import HandController

import cv2
import numpy as np
import tyro
import dataclasses
from typing import Optional

# Monkeypatch PyTorch loading locally to bypass 2.6 weights_only strictness without altering Libero source
import torch
_old_load = torch.load
torch.load = lambda *a, **k: _old_load(*a, **{**k, "weights_only": False})

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    seed: int = 7

def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    
    # Use OffScreenRenderEnv instead of ControlEnv to prevent standalone MuJoCo window hangs!
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language

def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loading Task suite: {args.task_suite_name}")
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    
    env, task_description = _get_libero_env(task, 256, args.seed)
    print(f"\nLoaded Task: {task_description}\n")
    
    # Initialize our HandController!
    print(">>> Connecting to Oculus Hand Tracking ...")
    teleop = HandController(pos_action_gain=5.0, rot_action_gain=2.0)
    
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    cv2.namedWindow("Live Hand Teleop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Hand Teleop", 600, 600)
    
    print("---------------------------------------------------------")
    print("READY TO DRIVE!")
    print("1. Hold your right hand PALM DOWN (mimicking the gripper orientation).")
    print("2. Pinch your right index finger to close the robotic gripper.")
    print("3. (Optional) Pinch your LEFT index to freeze ('clutch') movement.")
    print("Press Q to quit gracefully.")
    print("---------------------------------------------------------")

    # Countdown to give the user time to get into the correct pose
    for i in range(5, 0, -1):
        print(f"\r[Starting in {i}s... Get your hand PALM DOWN!]  ", end="", flush=True)
        time.sleep(1)
    print("\r[GO!] Control is now active.                         ")

    import pickle
    recorded_frames = []
    
    try:
        while True:
            action = teleop.predict(obs, wait_for_movement_enabled=False)
            
            # Record raw Hand transforms and SIM images for diagnostic review
            if teleop._state["poses"]:
                # MuJoCo/Gym images are stored from bottom up, so flip vertically!
                raw_img = obs["agentview_image"][::-1] 
                resized_img = cv2.resize(raw_img, (128, 128))
                
                recorded_frames.append({
                    "transformations": {k: v.copy() for k, v in teleop._state["poses"].items()},
                    "image": resized_img,
                    # Also store the post-mapped robot-space transform for the right hand
                    # so the visualizer can show both Oculus and Robot frames side-by-side.
                    "robot_transform": teleop.global_to_env_mat @ teleop._state["poses"]["r"] if "r" in teleop._state["poses"] else None,
                })
            
            # Diagnostics for telemetry debugging
            if not teleop._state["poses"]:
                print("\r[Status] Waiting for Oculus Tracking Data...", end="", flush=True)
            elif action is None:
                print("\r[Status] Clutched out (Frozen) ...          ", end="", flush=True)
            elif np.allclose(action[:6], 0):
                print("\r[Status] Hand tracked but stationary...       ", end="", flush=True)
            else:
                print(f"\r[Status] Driving >> Vel {np.linalg.norm(action[:3]):.3f}              ", end="", flush=True)
            
            if action is None:
                action = [0.0]*6 + [-1.0]
            
            # For testing disable robot rotation
            #action[3:] = 0.0

            obs, reward, done, info = env.step(action)
            
            # Show live camera feed for feedback
            img = np.ascontiguousarray(obs["agentview_image"][::-1])
            display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.putText(display_img, f"Task: {task_description}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("Live Hand Teleop", display_img)
            
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
                
            if done:
                print("Task Succeeded!")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()
        print(f"\n[Logging] Saving {len(recorded_frames)} tracked frames out to 'teleop_trace.pkl' ...")
        with open("teleop_trace.pkl", "wb") as f:
            pickle.dump(recorded_frames, f)
        print("Saved!")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
