#!/usr/bin/env python
"""
世界坐标映射脚本
使用外参实现2D图像到3D世界坐标的映射
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import sys

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认路径
DEFAULT_INTERNAL_CALIB_PATH = PROJECT_ROOT / "results" / "internal_paras" / "calibration.yaml"
DEFAULT_EXTERNAL_CALIB_PATH = PROJECT_ROOT / "results" / "external_paras" / "extrinsics.yaml"
DEFAULT_EXTERNAL_CONFIG_PATH = PROJECT_ROOT / "config" / "external_config.yaml"


class WorldCoordinateMapper:
    def __init__(self, fov_scale=0.6):
        """初始化世界坐标映射器

        Args:
            fov_scale: 鱼眼模型视场缩放因子 (0-1), 默认0.6
        """
        # 加载配置
        self.config = self._load_config()
        self.grid_spacing = self.config['mapping']['grid_spacing']
        self.default_height = self.config['mapping']['default_height']
        self.max_distance = self.config['mapping']['max_distance']

        # 加载内参
        self.internal_calib = self._load_calibration(DEFAULT_INTERNAL_CALIB_PATH, "内参")
        self.camera_matrix = np.array(self.internal_calib['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(self.internal_calib['distortion_coefficients']['data'])
        self.model = self.internal_calib.get('model', 'pinhole')
        self.image_width = self.internal_calib['image_width']
        self.image_height = self.internal_calib['image_height']

        # 计算新的相机矩阵（用于鱼眼去畸变）
        if self.model == 'fisheye':
            self.new_camera_matrix = self.camera_matrix.copy()
            self.new_camera_matrix[0, 0] *= fov_scale  # fx
            self.new_camera_matrix[1, 1] *= fov_scale  # fy
            self.new_camera_matrix[0, 2] = self.image_width / 2   # cx
            self.new_camera_matrix[1, 2] = self.image_height / 2  # cy
        else:
            self.new_camera_matrix = self.camera_matrix

        # 加载外参
        self.external_calib = self._load_calibration(DEFAULT_EXTERNAL_CALIB_PATH, "外参")
        self.rvec = np.array(self.external_calib['rotation_vector']).reshape(3, 1)
        self.tvec = np.array(self.external_calib['translation_vector']).reshape(3, 1)
        self.R = np.array(self.external_calib['rotation_matrix'])

        # 检查外参中是否保存了fov_scale，如果有则使用外参中的值
        if 'fov_scale' in self.external_calib and 'camera_matrix_used' in self.external_calib:
            saved_fov = self.external_calib['fov_scale']
            if abs(saved_fov - fov_scale) > 0.01:  # 如果差异较大，给出警告
                print(f"警告: 外参标定时使用的fov_scale={saved_fov}，当前使用的fov_scale={fov_scale}")
                print(f"建议使用相同的fov_scale以确保坐标准确性")
            # 使用外参标定时的相机矩阵
            self.calibration_camera_matrix = np.array(self.external_calib['camera_matrix_used'])
        else:
            # 兼容旧版本的外参文件
            self.calibration_camera_matrix = self.new_camera_matrix

        # 当前设置
        self.current_height = self.default_height
        self.show_grid = True
        self.measure_mode = False
        self.measure_points = []

        # 图像
        self.current_frame = None
        self.display_frame = None

    def _load_config(self) -> dict:
        """加载配置文件"""
        if not DEFAULT_EXTERNAL_CONFIG_PATH.exists():
            raise FileNotFoundError(f"配置文件不存在: {DEFAULT_EXTERNAL_CONFIG_PATH}")

        with open(DEFAULT_EXTERNAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_calibration(self, path: Path, name: str) -> dict:
        """加载标定文件"""
        if not path.exists():
            raise FileNotFoundError(f"{name}标定文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def image_to_world(self, u, v, Z):
        """将图像坐标转换为世界坐标

        Args:
            u, v: 图像坐标
            Z: 世界坐标系中的高度

        Returns:
            (X, Y, Z) 世界坐标，如果无法计算则返回None
        """
        # 使用标定时的相机矩阵（而非当前显示用的矩阵）
        K_inv = np.linalg.inv(self.calibration_camera_matrix)

        # 归一化坐标
        uv1 = np.array([[u], [v], [1]])
        xyz_c = K_inv @ uv1  # 相机坐标系中的方向向量

        # 相机坐标系到世界坐标系的转换
        # Pc = R * Pw + t
        # Pw = R^T * (Pc - t)
        R_inv = self.R.T

        # 对于给定的Z，求解X和Y
        # 这需要求解射线与Z平面的交点
        # 射线: Pc = lambda * xyz_c
        # 转换到世界坐标: Pw = R^T * (lambda * xyz_c - t)

        # 设定 Pw[2] = Z，求解lambda
        # R_inv @ (lambda * xyz_c - t) 的第3个分量 = Z
        # lambda * (R_inv @ xyz_c)[2] - (R_inv @ t)[2] = Z

        direction_w = R_inv @ xyz_c
        origin_w = -R_inv @ self.tvec

        if abs(direction_w[2]) < 1e-6:
            # 射线几乎平行于Z平面
            return None

        # 求解lambda
        lambda_val = (Z - origin_w[2]) / direction_w[2]

        if lambda_val < 0:
            # 点在相机后面
            return None

        # 计算世界坐标
        world_point = origin_w + lambda_val * direction_w

        X = float(world_point[0])
        Y = float(world_point[1])

        # 检查距离是否合理
        distance = np.sqrt(X**2 + Y**2)
        if distance > self.max_distance:
            return None

        return (X, Y, Z)

    def draw_grid(self, image):
        """在图像上绘制世界坐标网格"""
        # 创建网格点（在Z=0平面）
        grid_range = 5000  # 5米范围
        grid_points = []

        for x in range(-grid_range, grid_range + 1, self.grid_spacing):
            for y in range(-grid_range, grid_range + 1, self.grid_spacing):
                grid_points.append([float(x), float(y), 0.0])

        if not grid_points:
            return

        # 投影到图像
        grid_points_np = np.array(grid_points, dtype=np.float32)
        projected, _ = cv2.projectPoints(
            grid_points_np,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            None  # 已去畸变
        )

        # 绘制网格点
        for i, pt in enumerate(projected):
            x, y = int(pt[0][0]), int(pt[0][1])
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                # 原点用红色，其他用绿色
                if grid_points[i][0] == 0 and grid_points[i][1] == 0:
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                else:
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        # 绘制坐标轴
        axes_length = 1000  # 1米
        axes_points = np.array([
            [0, 0, 0],  # 原点
            [axes_length, 0, 0],  # X轴
            [0, axes_length, 0],  # Y轴
            [0, 0, axes_length]   # Z轴
        ], dtype=np.float32)

        axes_projected, _ = cv2.projectPoints(
            axes_points,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            None
        )

        # 绘制坐标轴
        origin = tuple(map(int, axes_projected[0][0]))
        x_axis = tuple(map(int, axes_projected[1][0]))
        y_axis = tuple(map(int, axes_projected[2][0]))
        z_axis = tuple(map(int, axes_projected[3][0]))

        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)  # X轴 - 红色
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)  # Y轴 - 绿色
        cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)  # Z轴 - 蓝色

        # 标注轴
        cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.measure_mode:
                # 测距模式：添加测量点
                world_coords = self.image_to_world(x, y, self.current_height)
                if world_coords:
                    self.measure_points.append((x, y, world_coords))
                    if len(self.measure_points) == 2:
                        # 计算距离
                        p1 = self.measure_points[0][2]
                        p2 = self.measure_points[1][2]
                        dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
                        print(f"\n测量距离: {dist:.1f} mm = {dist/1000:.3f} m")
                        print(f"  点1: ({p1[0]:.1f}, {p1[1]:.1f}, {p1[2]:.1f}) mm")
                        print(f"  点2: ({p2[0]:.1f}, {p2[1]:.1f}, {p2[2]:.1f}) mm")
                        self.measure_points = []

        elif event == cv2.EVENT_MOUSEMOVE:
            # 显示当前坐标
            world_coords = self.image_to_world(x, y, self.current_height)
            if world_coords:
                self.current_world_coords = world_coords
                self.current_image_coords = (x, y)
            else:
                self.current_world_coords = None

    def update_display(self):
        """更新显示"""
        if self.current_frame is None:
            return

        self.display_frame = self.current_frame.copy()

        # 绘制网格
        if self.show_grid:
            self.draw_grid(self.display_frame)

        # 显示当前坐标
        if hasattr(self, 'current_world_coords') and self.current_world_coords:
            x, y = self.current_image_coords
            X, Y, Z = self.current_world_coords

            # 画十字线
            cv2.line(self.display_frame, (x-10, y), (x+10, y), (255, 255, 0), 1)
            cv2.line(self.display_frame, (x, y-10), (x, y+10), (255, 255, 0), 1)

            # 显示坐标
            text = f"({X:.0f}, {Y:.0f}, {Z:.0f}) mm"
            cv2.putText(self.display_frame, text, (x+15, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示测量点
        if self.measure_mode and self.measure_points:
            for i, (px, py, world_pt) in enumerate(self.measure_points):
                cv2.circle(self.display_frame, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(self.display_frame, f"P{i+1}", (px+10, py-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示信息栏
        info_texts = [
            f"Height: {self.current_height:.0f} mm",
            f"Grid: {'ON' if self.show_grid else 'OFF'} (G)",
            f"Measure: {'ON' if self.measure_mode else 'OFF'} (M)",
            "H: Set height | ESC: Exit"
        ]

        for i, text in enumerate(info_texts):
            cv2.putText(self.display_frame, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def run_live(self):
        """实时相机模式"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

        window_name = "World Coordinate Mapper"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n操作说明:")
        print("  G - 显示/隐藏网格")
        print("  M - 切换测距模式")
        print("  H - 设置高度Z")
        print("  ESC - 退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 去畸变
            if self.model == 'fisheye':
                frame = cv2.fisheye.undistortImage(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs.reshape(-1, 1),
                    None,
                    self.new_camera_matrix
                )
            else:
                frame = cv2.undistort(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    self.new_camera_matrix
                )

            self.current_frame = frame
            self.update_display()
            cv2.imshow(window_name, self.display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('g'):
                self.show_grid = not self.show_grid
                print(f"网格: {'开启' if self.show_grid else '关闭'}")
            elif key == ord('m'):
                self.measure_mode = not self.measure_mode
                self.measure_points = []
                print(f"测距模式: {'开启' if self.measure_mode else '关闭'}")
            elif key == ord('h'):
                try:
                    height = float(input("输入高度Z (mm): "))
                    self.current_height = height
                    print(f"高度设置为: {height} mm")
                except ValueError:
                    print("输入无效")

        cap.release()
        cv2.destroyAllWindows()

    def run_image(self, image_path):
        """静态图像模式"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return

        # 去畸变
        if self.model == 'fisheye':
            image = cv2.fisheye.undistortImage(
                image,
                self.camera_matrix,
                self.dist_coeffs.reshape(-1, 1),
                None,
                self.new_camera_matrix
            )
        else:
            image = cv2.undistort(
                image,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.new_camera_matrix
            )

        self.current_frame = image

        window_name = "World Coordinate Mapper"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n操作说明:")
        print("  G - 显示/隐藏网格")
        print("  M - 切换测距模式")
        print("  H - 设置高度Z")
        print("  ESC - 退出")

        while True:
            self.update_display()
            cv2.imshow(window_name, self.display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('g'):
                self.show_grid = not self.show_grid
                print(f"网格: {'开启' if self.show_grid else '关闭'}")
            elif key == ord('m'):
                self.measure_mode = not self.measure_mode
                self.measure_points = []
                print(f"测距模式: {'开启' if self.measure_mode else '关闭'}")
            elif key == ord('h'):
                try:
                    height = float(input("输入高度Z (mm): "))
                    self.current_height = height
                    print(f"高度设置为: {height} mm")
                except ValueError:
                    print("输入无效")

        cv2.destroyAllWindows()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='世界坐标映射')
    parser.add_argument('--image', type=str, help='使用静态图像')
    parser.add_argument('--fov-scale', type=float, default=0.6,
                       help='鱼眼模型视场缩放 (0.3-1.0，默认: 0.6，值越小视场越大)')

    args = parser.parse_args()

    print("=" * 50)
    print("世界坐标映射系统")
    print("=" * 50)

    try:
        mapper = WorldCoordinateMapper(fov_scale=args.fov_scale)

        if args.image:
            mapper.run_image(args.image)
        else:
            mapper.run_live()

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保已完成：")
        print("1. 内参标定 (scripts/internal_paras/calibrate.py)")
        print("2. 外参标定 (scripts/external_paras/manual_extrinsic.py)")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()