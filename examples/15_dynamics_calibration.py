"""Onero A1 真机静态动力学参数辨识（重力参数）。

操作者在示教模式下把机械臂拖到不同构型，脚本切换到位置保持后采集静态
关节角和电机力矩，再用 Pinocchio 力矩回归器辨识重力相关参数。脚本不会
自动执行轨迹，也不会进行全行程摩擦扫描。

运行示例（右肩竖装）：
    python examples/15_dynamics_calibration.py \
        --model A1R --channel can0 --mounting humanoid_right \
        --output dynamics_cal.json

风险提示：
    - 运行前清空工作空间，确认急停可用，并由能够承托机械臂的人员操作。
    - 安装方向、夹爪和负载配置必须与真机一致，否则示教模式可能漂移。
    - 每次采样时必须松开机械臂，不能施加外力。
    - 结果只适用于标定时完全相同的机械臂、安装方向和末端负载组合。
    - 脚本结束前会要求先承托机械臂，再明确输入 DISABLE 失能；失能不是
      停止运动，会撤销保持力矩。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from onero_sdk import Arm, GravityCalibrator, OneroError


MOUNTING_CHOICES = (
    "upright",
    "a1_vertical",
    "side_mount_left",
    "side_mount_right",
    "humanoid_left",
    "humanoid_right",
    "ceiling",
)
MIN_CONFIGS = 20
STATE_SAMPLES = 25
STATE_SAMPLE_INTERVAL_S = 0.02
MAX_SAMPLE_VELOCITY_RAD_S = 0.02
MAX_SAMPLE_SPAN_RAD = 0.003
MIN_CONFIG_DISTANCE_RAD = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过真机多构型静态采样，辨识 Onero A1 重力相关动力学参数。",
    )
    parser.add_argument("--model", required=True, choices=("A1R", "A1L"))
    parser.add_argument("--channel", default="can0", help="SocketCAN 通道，默认 can0")
    parser.add_argument(
        "--mounting",
        required=True,
        choices=MOUNTING_CHOICES,
        help="真机安装方向；必须与实际方向一致",
    )
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="采样期间启用 Onero A1 夹爪重力补偿 profile",
    )
    parser.add_argument("--payload-mass", type=float, default=0.0, help="额外负载质量，kg")
    parser.add_argument(
        "--payload-cog",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="额外负载质心，相对末端坐标系，m",
    )
    parser.add_argument("--configs", type=int, default=40, help="计划采集构型数，默认 40")
    parser.add_argument("--settle-time", type=float, default=1.0, help="保持后稳定时间，默认 1 s")
    parser.add_argument("--damping", type=float, default=0.5, help="示教阻尼 kd，默认 0.5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dynamics_cal.json"),
        help="辨识结果路径，默认 ./dynamics_cal.json",
    )
    args = parser.parse_args()

    if args.configs < MIN_CONFIGS:
        parser.error(f"--configs 不能小于 {MIN_CONFIGS}")
    if not np.isfinite(args.settle_time) or args.settle_time < 0.2:
        parser.error("--settle-time 不能小于 0.2 s")
    if not np.isfinite(args.damping) or not 0.0 <= args.damping <= 5.0:
        parser.error("--damping 必须在 0 到 5 之间")
    if not np.isfinite(args.payload_mass) or args.payload_mass < 0.0:
        parser.error("--payload-mass 不能为负数")
    if not np.all(np.isfinite(args.payload_cog)):
        parser.error("--payload-cog 必须是有限数值")
    return args


def sample_static_state(arm: Arm) -> tuple[np.ndarray, np.ndarray, float, float]:
    """采集稳定状态，返回关节角、力矩、最大速度和最大关节摆动。"""
    joints = []
    torques = []
    max_velocity = 0.0
    for _ in range(STATE_SAMPLES):
        state = arm.get_state()
        joints.append(np.asarray(state.joints, dtype=float))
        torques.append(np.asarray(state.torques, dtype=float))
        max_velocity = max(max_velocity, max(abs(value) for value in state.velocities))
        time.sleep(STATE_SAMPLE_INTERVAL_S)

    joint_values = np.asarray(joints)
    torque_values = np.asarray(torques)
    if not np.all(np.isfinite(joint_values)) or not np.all(np.isfinite(torque_values)):
        raise RuntimeError("关节角或力矩反馈包含非有限数值")
    return (
        np.median(joint_values, axis=0),
        np.median(torque_values, axis=0),
        float(max_velocity),
        float(np.max(np.ptp(joint_values, axis=0))),
    )


def collect_static_samples(
    arm: Arm,
    count: int,
    settle_time: float,
    damping: float,
) -> tuple[list[list[float]], list[list[float]]] | None:
    configs: list[np.ndarray] = []
    torques: list[np.ndarray] = []

    arm.freedrive_enable(damping=damping)
    print("\n示教模式已启用。始终握持机械臂，缓慢改变肩、肘、腕构型。")
    print("避开关节极限、自碰撞、线缆拉扯和奇异构型。")
    print("命令：Enter=保持并采样，u=撤销上一条，d=提前解算，q=取消。")

    while len(configs) < count:
        command = input(f"\n构型 {len(configs) + 1}/{count}> ").strip().lower()
        if command == "q":
            return None
        if command == "u":
            if not configs:
                print("当前没有可撤销的样本。")
                continue
            configs.pop()
            torques.pop()
            print(f"已撤销，当前保留 {len(configs)} 个样本。")
            continue
        if command == "d":
            if len(configs) < MIN_CONFIGS:
                print(f"至少需要 {MIN_CONFIGS} 个构型，当前只有 {len(configs)} 个。")
                continue
            arm.freedrive_disable()
            break
        if command:
            print("未知命令；请输入 Enter、u、d 或 q。")
            continue
        if not arm.is_freedrive:
            raise RuntimeError("示教控制线程意外退出，请检查急停和 CAN 状态")

        arm.freedrive_disable()
        print(f"已切换到位置保持。松开机械臂，等待 {settle_time:.1f} s 后采样……")
        time.sleep(settle_time)
        q, tau, max_velocity, max_span = sample_static_state(arm)

        rejection = None
        if max_velocity > MAX_SAMPLE_VELOCITY_RAD_S:
            rejection = (
                f"最大关节速度 {max_velocity:.4f} rad/s，超过 "
                f"{MAX_SAMPLE_VELOCITY_RAD_S:.4f} rad/s"
            )
        elif max_span > MAX_SAMPLE_SPAN_RAD:
            rejection = (
                f"采样期间最大摆动 {max_span:.4f} rad，超过 "
                f"{MAX_SAMPLE_SPAN_RAD:.4f} rad"
            )
        elif configs:
            nearest = min(float(np.max(np.abs(q - previous))) for previous in configs)
            if nearest < MIN_CONFIG_DISTANCE_RAD:
                rejection = f"与已有构型过近（最近差值 {nearest:.3f} rad）"

        if rejection is None:
            configs.append(q)
            torques.append(tau)
            print(
                f"已记录 #{len(configs)}：q={np.round(q, 3).tolist()}，"
                f"tau={np.round(tau, 3).tolist()} Nm"
            )
        else:
            print(f"本次未记录：{rejection}。")

        if len(configs) >= count:
            break
        input("握稳机械臂并确认周围安全后，按 Enter 重新进入示教模式：")
        arm.freedrive_enable(damping=damping)

    return [value.tolist() for value in configs], [value.tolist() for value in torques]


def print_result(result) -> None:
    print("\n" + "=" * 64)
    print("静态动力学参数辨识完成")
    print(f"  有效构型数： {result.num_configs}")
    print(f"  静态可辨识秩： {result.rank}")
    print(f"  参数向量维度： {len(result.params)}")
    print(f"  力矩 RMS：   {result.residual_rms:.4f} Nm")
    print("  摩擦参数：   未辨识（本脚本不执行真机自动摩擦扫描）")


def disable_after_support(arm: Arm) -> None:
    """保持当前位置，直到操作者明确确认机械臂已受支撑。"""
    if not arm.enabled:
        return
    try:
        if arm.is_freedrive:
            arm.freedrive_disable()
        else:
            arm.stop(deceleration=2.0)
    except OneroError as exc:
        print(f"无法建立位置保持：{exc}")

    print("\n机械臂当前仍处于使能/保持状态。")
    while arm.enabled:
        try:
            confirmation = input(
                "确认已可靠承托机械臂后，输入 DISABLE 以失能并退出："
            ).strip()
        except KeyboardInterrupt:
            print("\n未失能；请先可靠承托机械臂。")
            continue
        except EOFError:
            print("\n输入已关闭，无法确认支撑；保留电机使能并仅断开 SDK 连接。")
            return
        if confirmation == "DISABLE":
            try:
                arm.disable()
            except OneroError as exc:
                print(f"失能失败：{exc}")
        else:
            print("输入不匹配，机械臂保持使能。")


def main() -> int:
    args = parse_args()
    if GravityCalibrator is None:
        print("缺少 pinocchio；请先安装 onero-sdk[dynamics] 或 onero-sdk[full]。")
        return 2
    if args.output.exists():
        confirmation = input(f"结果文件 {args.output} 已存在，输入 OVERWRITE 允许覆盖：")
        if confirmation.strip() != "OVERWRITE":
            print("已取消，未连接真机。")
            return 1
    if not args.output.parent.exists():
        print(f"输出目录不存在：{args.output.parent}")
        return 2

    print("=" * 64)
    print("Onero A1 真机静态动力学参数辨识")
    print(f"  机型/通道： {args.model} / {args.channel}")
    print(f"  安装方向：  {args.mounting}")
    print(f"  夹爪补偿：  {'开启' if args.with_gripper else '关闭'}（仅用于采样）")
    print(f"  额外负载：  {args.payload_mass:.3f} kg @ {list(args.payload_cog)} m")
    print(f"  计划构型：  {args.configs}")
    print("\n脚本不会自动运动，但会反复切换示教模式和位置保持。")
    print("标定期间不得改变夹爪、工具或负载，也不得在采样时触碰机械臂。")
    confirmation = input("确认工作空间、急停和承托条件安全后，输入 START：")
    if confirmation.strip() != "START":
        print("已取消，未连接真机。")
        return 1

    arm = None
    status = 0
    try:
        arm = Arm(
            args.model,
            channel=args.channel,
            mounting=args.mounting,
            with_gripper=args.with_gripper,
        )
        arm.enable()
        if args.payload_mass > 0.0:
            arm.set_payload(args.payload_mass, list(args.payload_cog))

        dynamics = arm._get_dynamics()
        calibrator = GravityCalibrator(dynamics._model, dynamics._data)
        print(f"重力方向（base）：{np.round(arm.gravity_direction, 3).tolist()}")
        print(f"起始关节角（rad）：{np.round(arm.get_state().joints, 4).tolist()}")
        input("握稳机械臂，准备开始拖动后按 Enter：")

        samples = collect_static_samples(
            arm,
            count=args.configs,
            settle_time=args.settle_time,
            damping=args.damping,
        )
        if samples is None:
            print("辨识已取消，未保存结果。")
            return 1

        configs, torques = samples
        print("\n正在解算静态动力学参数……")
        result = calibrator.fit(configs, torques)
        result.gravity_vector = arm.gravity_direction.tolist()
        if not np.all(np.isfinite(result.params)) or not np.isfinite(result.residual_rms):
            raise RuntimeError("辨识结果包含非有限数值，未保存")

        print_result(result)
        result.save(str(args.output))
        print(f"\n结果已保存：{args.output.resolve()}")
        print("下次必须使用相同安装方向和末端负载组合，并关闭额外重力 profile：")
        print(
            f'  arm = Arm("{args.model}", channel="{args.channel}", '
            f'mounting="{args.mounting}", with_gripper=False)'
        )
        print(f'  arm.load_calibration("{args.output.resolve()}")')
        print("  arm.enable()")
        print("不要再叠加 with_gripper 或 set_payload；末端负载变化后应重新辨识。")
        print("结果不会修改电机参数或写入电机固件。")
    except (KeyboardInterrupt, EOFError):
        print("\n用户取消辨识。")
        status = 130
    except (OneroError, ValueError, RuntimeError, ImportError) as exc:
        print(f"\n辨识失败：{exc}")
        status = 2
    finally:
        if arm is not None:
            try:
                disable_after_support(arm)
            finally:
                arm.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
