"""Onero A1 真机运动学标定（单点约束法）。

操作者在示教模式下，从不同关节构型让末端标定尖端触碰同一个固定参考点；
脚本只记录关节反馈，不会自动执行任何轨迹。

运行示例（右肩竖装）：
    python examples/16_kinematic_calibration.py \
        --model A1R --channel can0 --mounting humanoid_right \
        --tcp 0.000 0.000 0.150

    上述 TCP 仅演示数值格式；必须替换为 Link7 法兰原点到实际标定尖端的
    实测偏移，单位为米。

风险提示：
    - 运行前清空工作空间，确认急停可用，并由能够承托机械臂的人员操作。
    - ``--mounting``、TCP、夹爪和负载配置必须与真机一致，否则采样无效，
      示教模式也可能漂移。
    - 脚本结束前会要求先承托机械臂，再明确输入 DISABLE 失能；失能不是停止
      运动，会撤销保持力矩。
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from onero_sdk import Arm, KinematicCalibrator, OneroError


MOUNTING_CHOICES = (
    "upright",
    "a1_vertical",
    "side_mount_left",
    "side_mount_right",
    "humanoid_left",
    "humanoid_right",
    "ceiling",
)
MIN_TOUCHES = 15
SAMPLE_COUNT = 20
SAMPLE_INTERVAL_S = 0.01
MAX_SAMPLE_SPAN_RAD = 0.003
MIN_CONFIG_DISTANCE_RAD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过真机多构型触碰同一点，标定 Onero A1 运动学参数。",
    )
    parser.add_argument("--model", required=True, choices=("A1R", "A1L"))
    parser.add_argument("--channel", default="can0", help="SocketCAN 通道，默认 can0")
    parser.add_argument(
        "--mounting",
        required=True,
        choices=MOUNTING_CHOICES,
        help="真机安装方向；必须与机械臂实际方向一致",
    )
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="启用已标定的 Onero A1 夹爪重力补偿 profile",
    )
    parser.add_argument(
        "--tcp",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Link7 法兰原点到标定尖端的实测偏移，法兰坐标系，单位 m",
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
    parser.add_argument("--touches", type=int, default=20, help="计划采集次数，默认 20")
    parser.add_argument("--damping", type=float, default=0.5, help="示教阻尼 kd，默认 0.5")
    parser.add_argument(
        "--max-correction-mm",
        type=float,
        default=5.0,
        help="每个关节原点参数的最大修正量，默认 5 mm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kinematic_cal.json"),
        help="标定结果路径，默认 ./kinematic_cal.json",
    )
    args = parser.parse_args()

    if args.touches < MIN_TOUCHES:
        parser.error(f"--touches 不能小于 {MIN_TOUCHES}")
    if not np.isfinite(args.damping) or not 0.0 <= args.damping <= 5.0:
        parser.error("--damping 必须在 0 到 5 之间")
    if not np.isfinite(args.payload_mass) or args.payload_mass < 0.0:
        parser.error("--payload-mass 不能为负数")
    if not np.all(np.isfinite(args.payload_cog)):
        parser.error("--payload-cog 必须是有限数值")
    if not np.all(np.isfinite(args.tcp)):
        parser.error("--tcp 必须是有限数值")
    if not np.isfinite(args.max_correction_mm) or args.max_correction_mm <= 0.0:
        parser.error("--max-correction-mm 必须大于 0")
    return args


def sample_stable_joints(arm: Arm) -> tuple[np.ndarray, float]:
    """从示教反馈取中位数，并返回采样期间最大的关节摆动。"""
    samples = []
    for _ in range(SAMPLE_COUNT):
        samples.append(np.asarray(arm.state.joints, dtype=float))
        time.sleep(SAMPLE_INTERVAL_S)
    values = np.asarray(samples)
    q = np.median(values, axis=0)
    max_span = float(np.max(np.ptp(values, axis=0)))
    return q, max_span


def make_calibrator(arm: Arm, tcp: tuple[float, float, float]):
    """构造含标定尖端 TCP 的独立 FK，避免改动 Arm 当前的 FK。"""
    fk = copy.deepcopy(arm._get_fk())
    flange_to_tcp = np.eye(4)
    flange_to_tcp[:3, 3] = tcp
    fk._ee_fixed_transform = fk._ee_fixed_transform @ flange_to_tcp
    return KinematicCalibrator(fk)


def restore_touches(calibrator, touches: list[np.ndarray]) -> None:
    calibrator.clear()
    for q in touches:
        calibrator.record_touch(q=q.tolist())


def collect_touches(arm: Arm, calibrator, count: int, damping: float):
    touches: list[np.ndarray] = []

    arm.freedrive_enable(damping=damping)
    print("\n示教模式已启用。始终握持机械臂，缓慢改变肘部/腕部构型。")
    print("每次都让同一个末端标定尖端轻触同一个刚性参考点。")
    print("命令：Enter=记录，u=撤销上一条，d=提前解算，q=取消。")

    while len(touches) < count:
        command = input(f"\n触点 {len(touches) + 1}/{count}> ").strip().lower()
        if command == "q":
            return None
        if command == "u":
            if not touches:
                print("当前没有可撤销的触点。")
                continue
            touches.pop()
            restore_touches(calibrator, touches)
            print(f"已撤销，当前保留 {len(touches)} 个触点。")
            continue
        if command == "d":
            if len(touches) < MIN_TOUCHES:
                print(f"至少需要 {MIN_TOUCHES} 个触点，当前只有 {len(touches)} 个。")
                continue
            break
        if command:
            print("未知命令；请输入 Enter、u、d 或 q。")
            continue

        print("保持触点不动，正在采样反馈……")
        q, max_span = sample_stable_joints(arm)
        if max_span > MAX_SAMPLE_SPAN_RAD:
            print(
                f"本次未记录：采样期间最大摆动 {max_span:.4f} rad，"
                f"超过 {MAX_SAMPLE_SPAN_RAD:.4f} rad。请稳定后重试。"
            )
            continue

        if touches:
            nearest = min(float(np.max(np.abs(q - previous))) for previous in touches)
            if nearest < MIN_CONFIG_DISTANCE_RAD:
                print(
                    f"本次未记录：与已有构型过近（最近差值 {nearest:.3f} rad）。"
                    "请保持触碰同一点，但明显改变肘部或腕部构型。"
                )
                continue

        pos = calibrator.record_touch(q=q.tolist())
        touches.append(q)
        print(
            f"已记录 #{len(touches)}：FK=[{pos[0]:+.4f}, {pos[1]:+.4f}, "
            f"{pos[2]:+.4f}] m，当前分散度={calibrator.get_spread_mm():.2f} mm"
        )

    return calibrator, touches


def print_result(result, before_mm: float, max_correction_mm: float) -> None:
    print("\n" + "=" * 64)
    print("标定完成")
    print(f"  触点数：       {result.num_touches}")
    print(f"  标定前分散度： {before_mm:.3f} mm")
    print(f"  标定后残差：   {result.residual_mm:.3f} mm")
    print("  各关节原点修正（mm）：")
    for index, correction in enumerate(result.corrections, start=1):
        dx, dy, dz = correction * 1000.0
        print(f"    J{index}: dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f}")

    largest = float(np.max(np.abs(result.corrections))) * 1000.0
    if largest >= 0.99 * max_correction_mm:
        print("\n警告：至少一个修正量已碰到约束边界。不要直接使用该结果；")
        print("请增加构型差异、检查是否真的触碰同一点，然后重新标定。")


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
    if KinematicCalibrator is None:
        print("缺少 scipy；请先安装 onero-sdk[full] 或 scipy。")
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
    print("Onero A1 真机运动学标定")
    print(f"  机型/通道： {args.model} / {args.channel}")
    print(f"  安装方向：  {args.mounting}")
    print(f"  标定 TCP：  {list(args.tcp)} m（Link7 法兰坐标系）")
    print(f"  夹爪补偿：  {'开启' if args.with_gripper else '关闭'}")
    print(f"  额外负载：  {args.payload_mass:.3f} kg @ {list(args.payload_cog)} m")
    print(f"  计划触点：  {args.touches}")
    print("\n脚本不会自动运动，但会进入示教模式。")
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

        print(f"重力方向（base）：{np.round(arm.gravity_direction, 3).tolist()}")
        print(f"起始关节角（rad）：{np.round(arm.get_state().joints, 4).tolist()}")

        calibrator = make_calibrator(arm, tuple(args.tcp))
        collected = collect_touches(arm, calibrator, args.touches, args.damping)
        if collected is None:
            print("标定已取消，未保存结果。")
            return 1

        calibrator, _touches = collected
        arm.freedrive_disable()
        before_mm = calibrator.get_spread_mm()
        print("\n正在解算运动学修正……")
        result = calibrator.solve(max_correction_mm=args.max_correction_mm)
        print_result(result, before_mm, args.max_correction_mm)
        result.save(str(args.output))
        print(f"\n结果已保存：{args.output.resolve()}")
        print("下次创建 Arm 后可调用：")
        print(f'  arm.apply_kinematic_calibration("{args.output.resolve()}")')
        print("标定不会修改电机零点，也不会写入电机固件。")
        print("注意：当前 SDK 只把该结果应用到轻量 FK，不会自动修正 Pink IK 模型。")
    except (KeyboardInterrupt, EOFError):
        print("\n用户取消标定。")
        status = 130
    except (OneroError, ValueError, RuntimeError) as exc:
        print(f"\n标定失败：{exc}")
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
