import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 1  # For dynamic teleop, we want high responsiveness

    task_suite_name: str = "libero_spatial"
    task_id: Optional[int] = None
    prompt: Optional[str] = None
    num_steps_wait: int = 10
    
    seed: int = 7
    render: bool = False  # Keep false, we use cv2 to render interactions!


def interactive_teleop(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 500  # Give more time for interactive play
    elif args.task_suite_name == "libero_object":
        max_steps = 500
    elif args.task_suite_name == "libero_goal":
        max_steps = 500
    elif args.task_suite_name == "libero_10":
        max_steps = 600
    elif args.task_suite_name == "libero_90":
        max_steps = 600
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task_ids = [args.task_id] if args.task_id is not None else range(task_suite.n_tasks)
    
    cv2.namedWindow("Interactive VLA Teleop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interactive VLA Teleop", 600, 600)
    
    print("\n--- Interactive VLA Teleop Controls ---")
    print("W : Follow VLA predictive action (Forward on Dynamic Axis)")
    print("S : Reverse VLA predictive action (Backward on Dynamic Axis)")
    print("A : Move arm left (Orthogonal deviation)")
    print("D : Move arm right (Orthogonal deviation)")
    print("Q : Quit current episode")
    print("No key : Stop movement (Zero action, hold gripper)")
    print("---------------------------------------\n")

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.render)

        active_prompt = args.prompt if args.prompt is not None else str(task_description)
        
        # Only do 1 trial interactively to allow user to try different tasks
        for episode_idx in range(1):
            logging.info(f"\nTask: {active_prompt}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocess observations
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img_processed = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_processed = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    element = {
                        "observation/image": img_processed,
                        "observation/wrist_image": wrist_processed,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": args.prompt if args.prompt is not None else str(task_description),
                    }

                    # Always replan at every step to get the freshest VLA prediction
                    action_chunk = client.infer(element)["actions"]
                    vla_action = action_chunk[0]  # The immediate next action predicted by VLA

                    # Interactive CV2 Loop
                    # Display original hi-res image for better user experience
                    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Add current predicted action overlay to the image
                    cv2.putText(display_img, f"Task: {active_prompt}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.imshow("Interactive VLA Teleop", display_img)
                    
                    key = cv2.waitKey(50) & 0xFF  # Wait 50ms for keypress
                    
                    # Construct executed action based on user input
                    executed_action = np.zeros(7)
                    executed_action[-1] = vla_action[-1]  # ALWAYS preserve VLA's prediction for the gripper state
                    
                    if key == ord('w'):
                        # Follow VLA intent
                        executed_action = vla_action
                    elif key == ord('s'):
                        # Reverse VLA intent (spatial dimensions only)
                        executed_action[:6] = -vla_action[:6]
                    elif key == ord('a'):
                        # Hardcoded orthogonal movement (e.g., left)
                        executed_action[1] = 0.5  # Move purely in +Y (relative to base)
                    elif key == ord('d'):
                        # Hardcoded orthogonal movement (e.g., right)
                        executed_action[1] = -0.5 # Move purely in -Y (relative to base)
                    elif key == ord('q'):
                        logging.info("Quitting episode early...")
                        break
                    else:
                        # No movement if no key is pressed (gripper maintains state)
                        pass

                    obs, reward, done, info = env.step(executed_action.tolist())
                    
                    if done:
                        logging.info("Task Succeeded!")
                        break
                        
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

        try:
            env.close()
        except Exception:
            pass
        finally:
            del env

    cv2.destroyAllWindows()


def _get_libero_env(task, resolution, seed, render=False):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    if render:
        env = ControlEnv(has_renderer=True, has_offscreen_renderer=True, **env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    interactive_teleop(args)
