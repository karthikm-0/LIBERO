import sys
import os
import time
import pathlib
import logging

import cv2
import numpy as np
import tyro
import dataclasses
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sys.path.append("/home/karthikm/robot-lightning")
from robots.hand import HandController

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
    output_video: str = "debug_recording.mp4"
    max_steps: int = 200

def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language

def plot_transform(ax, transform, name):
    t = transform[:3, 3]
    x_axis = transform[:3, 0]
    y_axis = transform[:3, 1]
    z_axis = transform[:3, 2]
    scale = 0.2
    ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, normalize=True)
    ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, normalize=True)
    ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, normalize=True)
    ax.text(t[0], t[1], t[2], name)

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    
    env, task_description = _get_libero_env(task, 400, args.seed)
    
    print(">>> Connecting to Oculus Hand Tracking ...")
    teleop = HandController()
    
    env.reset()
    obs = env.set_init_state(initial_states[0])

    print(f"===========================================================")
    print(f"Recording {args.max_steps} frames WITHOUT Lag...")
    print("Move your hand around distinctly (e.g., straight forward, then straight right).")
    print("Press Q or wait for the steps to run out to exit.")
    print(f"===========================================================\n")
    
    # Store history to render OFFLINE so we don't lag the physics thread!
    frame_history = []
    
    try:
        for step in range(args.max_steps):
            action = teleop.predict(obs, wait_for_movement_enabled=False)
            if action is None:
                action = [0.0]*6 + [-1.0]

            # Grab raw pose safely
            raw_pose = None
            poses = teleop._state.get("poses", {})
            if 'r' in poses:
                raw_pose = poses['r'].copy()

            # Execute on robot
            obs, reward, done, info = env.step(action)
            
            sim_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            sim_img = cv2.cvtColor(sim_img, cv2.COLOR_RGB2BGR)
            
            frame_history.append((sim_img, raw_pose))
            
            # Live feedback
            cv2.imshow("Teleop Recording (Smooth)...", sim_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        env.close()
        cv2.destroyAllWindows()
        teleop._running = False
        
    print(f"\nTeleop Finished! Captured {len(frame_history)} frames.")
    print("Rendering Matplotlib Diagnostic Video OFFLINE... Please wait (this might take 10 seconds).")
    
    # Offline rendering
    fig = plt.figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    video_writer = None
    
    for i, (sim_img, raw_pose) in enumerate(frame_history):
        ax.clear()
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X (Red)')
        ax.set_ylabel('Y (Green)')
        ax.set_zlabel('Z (Blue)')
        
        plot_transform(ax, np.eye(4), 'World Origin')
        if raw_pose is not None:
            plot_transform(ax, raw_pose, 'Raw Right Hand')

        canvas.draw()
        raw_rgba = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = raw_rgba.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        h, w = plot_img.shape[:2]
        sim_img_resized = cv2.resize(sim_img, (h, h))
        
        combined_img = np.hstack((sim_img_resized, plot_img))
        cv2.putText(combined_img, f"Frame {i}/{len(frame_history)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        if video_writer is None:
            h_c, w_c = combined_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output_video, fourcc, 15.0, (w_c, h_c))
        
        video_writer.write(combined_img)
        print(f"\rRendering Frame {i+1}/{len(frame_history)} ...", end="")

    if video_writer is not None:
        video_writer.release()
    plt.close(fig)
    print(f"\nSaved {args.output_video} successfully! Play it back to verify the axes!")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
