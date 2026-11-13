"""movep 笛卡尔点到点运动示例（默认 MuJoCo 仿真）。

运行：
    python3 examples/30_basic_movep.py
    python3 examples/30_basic_movep.py --real --channel can0

真机风险提示：本示例会先经过关节零位，再面向人执行约 16 cm × 8 cm 的轨迹。
movep 只约束终点，不保证末端沿直线移动；运行前必须确认机械臂整个运动范围
内无人、无障碍物且急停可用。未经具体真机环境验证，请先在仿真中检查。
"""

import argparse

import numpy as np

from onero_sdk import Arm


MODEL = "A1R"
MOUNTING = "humanoid_right"
# 人形右肩安装下，末端位于人体正面（world +X）的非奇异准备位。
READY_JOINTS = [0.0, -0.5, -0.3, 1.0, 0.0, -0.5, 0.0]
WAYPOINTS = [
    ("右下", [0.0, -0.08, 0.0]),
    ("右上", [0.0, -0.08, 0.08]),
    ("中上", [0.0, 0.0, 0.08]),
    ("左上", [0.0, 0.08, 0.08]),
    ("左下", [0.0, 0.08, 0.0]),
    ("回到起点", [0.0, 0.0, 0.0]),
]


def main():
    parser = argparse.ArgumentParser(description="Onero A1 movep 点到点轨迹演示")
    parser.add_argument("--real", action="store_true", help="连接真机（默认 MuJoCo 仿真）")
    parser.add_argument("--channel", default="can0", help="真机 SocketCAN 接口（默认 can0）")
    args = parser.parse_args()

    viewer = None
    if args.real:
        print("风险提示：即将连接 A1R 人形右肩真机。")
        print("movep 的点间路径不是直线，请确认完整运动范围无障碍且急停可用。")
        input("确认机械臂已可靠安装、轨迹安全，按 Enter 继续（Ctrl+C 取消）...")
        arm = Arm(MODEL, channel=args.channel, mounting=MOUNTING, world_frame=True)
    else:
        try:
            import mujoco.viewer
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError as exc:
            raise SystemExit(
                '缺少仿真或运动学依赖，请运行: pip install -e ".[sim,ik]"'
            ) from exc

        backend = MujocoBackend(model_name=MODEL, realtime=True)
        arm = Arm(MODEL, backend=backend, mounting=MOUNTING, world_frame=True)
        viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
        backend._on_step = viewer.sync

    try:
        arm.enable()

        print("1/3 回到关节零位")
        arm.home(duration=3.0)

        print("2/3 移动到非奇异起始位置")
        arm.movej(READY_JOINTS, duration=3.0)
        start_position, start_rotation = arm.get_pose(frame="world")

        print("3/3 执行 movep 闭合轨迹")
        for name, offset in WAYPOINTS:
            target = start_position + np.asarray(offset)
            print(f"  {name}: {target.round(3).tolist()} m")
            arm.movep(
                position=target,
                rotation=start_rotation,
                duration=2.5,
                frame="world",
            )

        if args.real:
            print("轨迹完成，正在回到关节零位。")
            arm.home(duration=3.0)
            print("已回到关节零位；机械臂保持使能。可靠支撑后再主动失能。")
        else:
            print("MuJoCo 轨迹完成。")
    except BaseException:
        if args.real and arm.enabled:
            arm.emergency_stop()
            print("真机已紧急停止并失能，请立即支撑机械臂。")
        raise
    finally:
        if not args.real and arm.enabled:
            arm.disable()
        arm.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
