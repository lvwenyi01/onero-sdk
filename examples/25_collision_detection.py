"""碰撞检测演示：默认 MuJoCo 仿真，--real 切换 A1R 右肩真机。

仿真会自动注入一次力矩冲击；真机请只用固定的柔软泡棉障碍物测试。

    python examples/25_collision_detection.py
    python examples/25_collision_detection.py --real --channel can0
    python examples/25_collision_detection.py --real --joint 3 --amplitude 0.15
    python examples/25_collision_detection.py --real --impact-threshold 80
"""

import argparse
import math
import os
import sys

# ROS Humble's Pinocchio conflicts with the venv Pinocchio/NumPy ABI.
env = os.environ.copy()
restart = False
for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
    paths = env.get(name, "").split(os.pathsep)
    cleaned = os.pathsep.join(path for path in paths if not path.startswith("/opt/ros"))
    if cleaned != env.get(name, ""):
        env[name] = cleaned
        restart = True
if restart:
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

from onero_sdk import Arm, CollisionConfig, CollisionResponse


parser = argparse.ArgumentParser(description="A1R 碰撞检测演示")
parser.add_argument("--real", action="store_true", help="连接真机（默认仿真）")
parser.add_argument("--channel", default="can0", help="SocketCAN 接口")
parser.add_argument("--joint", type=int, choices=range(1, 8), default=2,
                    help="运动的关节 J1-J7（默认 J2）")
parser.add_argument("--amplitude", type=float, default=1.0,
                    help="运动幅度，单位 rad（默认 1.0）")
parser.add_argument("--impact-threshold", type=float, default=80.0,
                    help="冲击通道阈值（默认 80）")
parser.add_argument("--energy-threshold", type=float, default=25.0,
                    help="能量通道阈值（默认 25）")
args = parser.parse_args()
for name in ("amplitude", "impact_threshold", "energy_threshold"):
    if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
        parser.error(f"--{name.replace('_', '-')} 必须是大于 0 的有限数")

mounting = "humanoid_right"
viewer = None
sim = {"inject_at": None, "injected": False}

if args.real:
    arm = Arm("A1R", channel=args.channel, mounting=mounting)
else:
    try:
        import mujoco.viewer
        from onero_sdk.mujoco_backend import MujocoBackend
    except ImportError:
        raise SystemExit('仿真依赖未安装，请运行: pip install -e ".[sim,dynamics]"')

    backend = MujocoBackend(model_name="A1R", realtime=True)
    viewer = mujoco.viewer.launch_passive(backend._model, backend._data)

    def update_sim():
        viewer.sync()
        if (sim["inject_at"] is not None and not sim["injected"]
                and backend._step_count >= sim["inject_at"]):
            backend._data.qfrc_applied[args.joint - 1] += 12.0
            sim["injected"] = True

    backend._on_step = update_sim
    arm = Arm("A1R", backend=backend, mounting=mounting)

clean_sim_exit = False
try:
    if args.real:
        print("真机将直接启动：A1R 右肩垂直安装，请确保机械臂已支撑。")

    try:
        initial = arm.enable()
        arm.set_stiffness("medium")

        def on_collision(event):
            print(f"\n>>> 碰撞触发: {event.channel}, J{event.joint_index + 1}, "
                  f"强度 {event.severity:.1f}x")

        arm.enable_collision_detection(CollisionConfig(
            # 抗噪演示起点，不是硬件安全额定值。
            impact_threshold=args.impact_threshold,
            energy_threshold=args.energy_threshold,
            min_velocity=0.05,
            warmup_frames=100,
            response=CollisionResponse.BOTH,
        ), on_collision=on_collision)
        print("碰撞检测已启用: "
              f"impact={args.impact_threshold:g}, energy={args.energy_threshold:g}")

        target = list(initial.joints)
        index = args.joint - 1
        lower, upper = arm.get_joint_limits()[index]
        if target[index] + args.amplitude <= upper:
            delta = args.amplitude
        elif target[index] - args.amplitude >= lower:
            delta = -args.amplitude
        else:
            raise SystemExit(f"J{args.joint} 剩余行程不足 {args.amplitude} rad")
        target[index] += delta
        print(f"J{args.joint} 将运动 {delta:+.3f} rad")

        if not args.real:
            sim["inject_at"] = backend._step_count + 320
            print("仿真将在约 1.6 秒后注入力矩冲击...")

        state = arm.movej(target, duration=5.0)
        event = arm.last_collision
        if event:
            print(f"机械臂已停止: {event.magnitude:.1f} > {event.threshold:.1f}")
            print("停止位置:", [round(float(q), 3) for q in state.joints])
        else:
            print("运动完成，未检测到碰撞。")
            if args.real:
                print("若泡棉接触仍未触发，可小幅降低 --impact-threshold；"
                      "每次调整后先做无障碍运动确认不误报。")

        if not args.real and event is None:
            raise RuntimeError("仿真冲击未触发碰撞检测")

        arm.disable()
        print("演示结束，机械臂已自动失能。")
    except BaseException:
        if args.real and arm.enabled:
            arm.emergency_stop()
            print("真机已急停；电机已失能，请继续支撑机械臂。")
        raise
    clean_sim_exit = viewer is not None
finally:
    if arm.enabled and not args.real:
        arm.disable()
    arm.close()
    if viewer is not None:
        viewer.close()

if clean_sim_exit:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
