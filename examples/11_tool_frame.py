"""
工具坐标系 (TCP) — 夹爪/灵巧手的坐标变换。

当末端安装了夹爪、灵巧手或其他工具时，需要设置 TCP 偏移，
使所有运动指令以工具尖端(而非法兰面)为参考点。

概念:
  法兰面 (flange):  机械臂最后一个关节输出的参考面
  TCP (Tool Center Point): 工具的实际作用点
  TCP 偏移:  从法兰面到 TCP 的变换 [x, y, z, rx, ry, rz]

    T_base_tcp = T_base_flange × T_flange_tcp
                 (FK 输出)       (用户设置)

示例场景:
  - 夹爪尖端离法兰面 17cm → tcp_offset = [0, 0, 0.17, 0, 0, 0]
  - 工具安装有角度 → tcp_offset = [0, 0, 0.1, 0, 0.3, 0]

运行:
    python examples/13_tool_frame.py
"""

import math
import time
import numpy as np
from onero_sdk import Arm

TCP_OFFSET = np.array([0.0, 0.0, 0.17])  # 按实际工具尺寸修改

arm = Arm("A1R", channel="can0", mounting="humanoid_right", with_gripper=True)
arm.enable()

try:
    # 先离开全零关节位形，避免从肩肘奇异点启动笛卡尔运动。
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

    # === 1. 设置 TCP 偏移 ===
    print("1. 设置 TCP (工具坐标系偏移)")
    print()
    print("   假设末端安装了一个 17cm 的平行夹爪:")
    print("   夹爪尖端在法兰面 Z 轴正方向 0.17m 处")
    print()

    # TCP 偏移:
    # position: [x, y, z] 法兰面坐标系下的平移 (m)
    # rotation: 旋转向量 [rx, ry, rz] (rad), 可选
    # 同一关节姿态下对比 flange 与 TCP；set_tcp() 本身不会引起运动。
    arm.clear_tcp()
    flange_pos, flange_rot = arm.get_pose()
    arm.set_tcp(position=TCP_OFFSET)
    tcp_pos, _ = arm.get_pose()
    expected_tcp_pos = flange_pos + flange_rot @ TCP_OFFSET

    print("   法兰/TCP 静态对比:")
    print(f"   法兰位置:     {flange_pos.round(4).tolist()}")
    print(f"   TCP 位置:     {tcp_pos.round(4).tolist()}")
    print(f"   TCP 理论位置: {expected_tcp_pos.round(4).tolist()}")
    print(f"   两参考点距离: {np.linalg.norm(tcp_pos - flange_pos) * 1000:.1f} mm")
    print(f"   变换计算误差: {np.linalg.norm(tcp_pos - expected_tcp_pos) * 1000:.6f} mm")
    print(f"   TCP 设置完成: position=[0, 0, 0.17]")
    print(f"   现在 movel 的目标位置是夹爪尖端的位置，不是法兰面!")

    # === 2. 使用 TCP 后的运动 ===
    print("\n2. 使用 TCP 运动")
    start_tcp_pos, start_tcp_rot = arm.get_pose()
    target_tcp_pos = start_tcp_pos + np.array([0.02, 0.0, 0.0])
    print(f"   让夹爪尖端沿 base +X 平移 2cm，到达 {target_tcp_pos.round(3).tolist()}:")
    arm.movel(position=target_tcp_pos, rotation=start_tcp_rot, duration=4.0)
    time.sleep(1.0)
    arm.get_state()  # 主动刷新稳定后的真实关节反馈
    pos, rot = arm.get_pose()
    print(f"   末端位姿 (TCP): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"   TCP 实际位移: {((pos - start_tcp_pos) * 1000).round(2).tolist()} mm")
    print(f"   TCP 目标误差: {np.linalg.norm(pos - target_tcp_pos) * 1000:.2f} mm")
    # 注: 法兰面位于 TCP 沿工具 -Z 方向 17cm 处。

    # === 3. 不同工具的 TCP 设置举例 ===
    print("\n3. 常见工具的 TCP 偏移示例:")
    print()
    print("   (a) 平行夹爪 (直装, 长 17cm):")
    print("       arm.set_tcp(position=[0, 0, 0.17])")
    print()
    print("   (b) 灵巧手 (长 20cm, 指尖作为 TCP):")
    print("       arm.set_tcp(position=[0, 0, 0.20])")
    print()
    print("   (c) 吸盘 (45° 安装角, 长 10cm):")
    print("       arm.set_tcp(position=[0, 0, 0.10], rotation=[0, 0.785, 0])")
    print()
    print("   (d) 焊枪 (偏心安装 + 倾斜):")
    print("       arm.set_tcp(position=[0.05, 0, 0.12], rotation=[0, 0.52, 0])")

    # === 4. 多工具切换 ===
    print("\n4. 多工具快速切换")

    TOOLS = {
        "gripper": {"position": [0.0, 0.0, 0.17]},
        "camera":  {"position": [0.0, 0.05, 0.08]},
        "suction": {"position": [0.0, 0.0, 0.10], "rotation": [0.0, math.pi/4, 0.0]},
    }

    for name, tcp_args in TOOLS.items():
        arm.set_tcp(**tcp_args)
        print(f"   切换到 '{name}': {tcp_args}")
        # 同一个 movel 目标, TCP 不同则法兰面位置不同
        # arm.movel(position=[0.3, 0.0, 0.17, 0.0, 0.0, 0.0], duration=1.5)

    # === 5. 重置 TCP (回到法兰面) ===
    print("\n5. 重置 TCP (法兰面为参考)")
    arm.clear_tcp()
    print("   TCP 已重置，运动指令以法兰面为参考")

    # === 6. TCP 的坐标变换公式 ===
    print("\n6. 坐标变换关系:")
    print()
    print("   T_base_tcp = T_base_flange × T_flange_tcp")
    print()
    print("   其中:")
    print("   T_base_flange = FK(joints)        ← 正运动学计算")
    print("   T_flange_tcp  = set_tcp() 设定    ← 用户配置")
    print("   T_base_tcp    = arm.get_pose()    ← SDK 返回 (含 TCP 偏移)")
    print()
    print("   知道任意两个, 可以求第三个:")
    print("   T_flange_tcp  = T_base_flange⁻¹ × T_base_tcp")
    print("   T_base_flange = T_base_tcp × T_flange_tcp⁻¹")

finally:
    arm.disable()