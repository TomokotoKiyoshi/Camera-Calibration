#!/usr/bin/env python
"""
手动外参标定脚本
通过手动选择地砖角点建立相机外参
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
DEFAULT_EXTERNAL_CONFIG_PATH = PROJECT_ROOT / "config" / "external_config.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "external_paras" / "extrinsics.yaml"


class ManualExtrinsicCalibrator:
    def __init__(self, image_path: str = None, fov_scale: float = 0.6):
        """初始化手动外参标定器

        Args:
            image_path: 用于标定的图像路径，如果为None则使用摄像头
            fov_scale: 鱼眼模型视场缩放因子 (0-1), 默认0.6
        """
        # 加载配置
        self.config = self._load_config()
        self.tile_size = self.config['manual_calibration']['tile_size']

        # 加载内参
        self.internal_calib = self._load_internal_calibration()
        self.camera_matrix = np.array(self.internal_calib['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(self.internal_calib['distortion_coefficients']['data'])
        self.model = self.internal_calib.get('model', 'pinhole')

        # 计算新的相机矩阵（用于鱼眼去畸变）
        if self.model == 'fisheye':
            self.new_camera_matrix = self.camera_matrix.copy()
            self.new_camera_matrix[0, 0] *= fov_scale  # fx
            self.new_camera_matrix[1, 1] *= fov_scale  # fy
            self.new_camera_matrix[0, 2] = self.internal_calib['image_width'] / 2   # cx
            self.new_camera_matrix[1, 2] = self.internal_calib['image_height'] / 2  # cy
        else:
            self.new_camera_matrix = self.camera_matrix

        # 初始化点集
        self.image_points = []  # 2D图像点
        self.world_points = []  # 3D世界点

        # 显示参数
        self.point_size = 10
        self.line_thickness = 2
        self.font_scale = 0.8

        # 图像
        self.image_path = image_path
        self.original_image = None
        self.display_image = None
        self.image_width = self.internal_calib['image_width']
        self.image_height = self.internal_calib['image_height']

    def _load_config(self) -> dict:
        """加载外参配置文件"""
        config_path = DEFAULT_EXTERNAL_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_internal_calibration(self) -> dict:
        """加载内参标定结果"""
        calib_path = DEFAULT_INTERNAL_CALIB_PATH
        if not calib_path.exists():
            raise FileNotFoundError(f"内参标定文件不存在: {calib_path}\n请先运行内参标定")

        with open(calib_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_image(self):
        """加载或采集图像"""
        if self.image_path:
            # 从文件加载
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError(f"无法读取图像: {self.image_path}")
        else:
            # 从摄像头采集
            print("按空格键采集图像...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头")
                    sys.exit(1)

                cv2.imshow("Press SPACE to capture", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    self.original_image = frame
                    cv2.destroyAllWindows()
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

            cap.release()

        # 去畸变
        if self.model == 'fisheye':
            self.original_image = cv2.fisheye.undistortImage(
                self.original_image,
                self.camera_matrix,
                self.dist_coeffs.reshape(-1, 1),
                None,
                self.new_camera_matrix
            )
        else:
            self.original_image = cv2.undistort(
                self.original_image,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.new_camera_matrix
            )

        self.display_image = self.original_image.copy()

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击：添加点
            self.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击：删除最后一个点
            self.remove_last_point()
        elif event == cv2.EVENT_MOUSEMOVE:
            # 显示坐标
            self.update_display(mouse_pos=(x, y))

    def add_point(self, x, y):
        """添加标定点"""
        self.image_points.append([x, y])
        self.world_points.append([0, 0, 0])  # 占位符，稍后自动计算

        idx = len(self.image_points) - 1
        print(f"点{idx+1} 已添加 - 图像坐标: ({x}, {y})")

        self.update_display()

    def remove_last_point(self):
        """删除最后一个点"""
        if self.image_points:
            self.image_points.pop()
            self.world_points.pop()
            print(f"删除最后一个点，剩余 {len(self.image_points)} 个点")
            self.update_display()

    def update_display(self, mouse_pos=None):
        """更新显示"""
        self.display_image = self.original_image.copy()

        # 绘制已选择的点
        for i, (img_pt, world_pt) in enumerate(zip(self.image_points, self.world_points)):
            # 画点
            cv2.circle(self.display_image, tuple(map(int, img_pt)),
                      self.point_size, (0, 0, 255), -1)

            # 标注编号和坐标
            text = f"{i+1}"
            cv2.putText(self.display_image, text,
                       (int(img_pt[0] + 15), int(img_pt[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (0, 255, 0), self.line_thickness)


            # 连线（如果有前一个点）
            if i > 0:
                prev_pt = self.image_points[i-1]
                cv2.line(self.display_image,
                        tuple(map(int, prev_pt)),
                        tuple(map(int, img_pt)),
                        (0, 255, 0), self.line_thickness)

        # 显示鼠标位置
        if mouse_pos:
            cv2.putText(self.display_image, f"({mouse_pos[0]}, {mouse_pos[1]})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 1)

        # 显示统计信息
        info_text = f"Points: {len(self.image_points)} | Left: Add | Right: Remove | Enter: Calculate | ESC: Quit"
        cv2.putText(self.display_image, info_text,
                   (10, self.image_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)

    def input_world_coordinates(self):
        """自动计算世界坐标"""
        print("\n" + "="*40)
        print("自动计算世界坐标")
        print("="*40)

        num_points = len(self.image_points)
        print(f"已选择 {num_points} 个点")

        # 询问网格列数
        cols = int(input("每行有几个点？: "))
        rows = (num_points + cols - 1) // cols  # 向上取整

        print(f"网格大小: {cols}x{rows}")
        print(f"地砖尺寸: {self.tile_size[0]}x{self.tile_size[1]} mm")
        print("\n计算结果:")

        # 自动计算每个点的世界坐标
        for i in range(num_points):
            row = i // cols
            col = i % cols
            world_x = col * self.tile_size[0]
            world_y = row * self.tile_size[1]
            self.world_points[i] = [world_x, world_y, 0]
            print(f"  点{i+1}: ({world_x:.0f}, {world_y:.0f}, 0) mm")

    def calculate_extrinsics(self):
        """计算外参"""
        if len(self.image_points) < 4:
            print("错误：至少需要4个点")
            return None, None

        # 先输入世界坐标
        self.input_world_coordinates()

        # 转换为numpy数组
        obj_points = np.array(self.world_points, dtype=np.float32)
        img_points = np.array(self.image_points, dtype=np.float32)

        # 使用solvePnP计算外参（使用去畸变后的相机矩阵）
        if self.model == 'fisheye':
            # 鱼眼模型：使用new_camera_matrix
            ret, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                self.new_camera_matrix, None,  # 使用去畸变后的相机矩阵
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            ret, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                self.new_camera_matrix, None,  # 使用去畸变后的相机矩阵
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not ret:
            print("外参计算失败")
            return None, None

        # 转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 计算重投影误差（使用相同的相机矩阵）
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec,
                                        self.new_camera_matrix, None)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(img_points - projected, axis=1)
        mean_error = np.mean(errors)

        print(f"\n标定结果:")
        print(f"  平均重投影误差: {mean_error:.2f} 像素")
        print(f"  旋转向量:\n{rvec.T}")
        print(f"  平移向量:\n{tvec.T}")

        return rvec, tvec

    def save_results(self, rvec, tvec, fov_scale):
        """保存外参结果"""
        # 确保输出目录存在
        output_dir = DEFAULT_OUTPUT_PATH.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 构建结果字典
        from datetime import datetime
        results = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_points': len(self.image_points),
            'tile_size': self.tile_size,
            'model': self.model,
            'fov_scale': fov_scale if self.model == 'fisheye' else 1.0,
            'camera_matrix_used': self.new_camera_matrix.tolist(),  # 保存实际使用的相机矩阵
            'rotation_vector': rvec.flatten().tolist(),
            'translation_vector': tvec.flatten().tolist(),
            'rotation_matrix': R.tolist(),
            'image_points': self.image_points,  # 已经是list
            'world_points': self.world_points
        }

        # 保存YAML
        with open(DEFAULT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)

        print(f"\n外参已保存至: {DEFAULT_OUTPUT_PATH}")

    def run(self, fov_scale):
        """运行标定流程

        Args:
            fov_scale: 传递进来的fov_scale参数
        """
        print("=" * 50)
        print("手动外参标定")
        print("=" * 50)
        print(f"地砖尺寸: {self.tile_size[0]}x{self.tile_size[1]} mm")
        if self.model == 'fisheye':
            print(f"鱼眼视场缩放: {fov_scale}")
        print("\n操作说明:")
        print("  1. 按顺序点击地砖角点（从左到右，从上到下）")
        print("  2. 按Enter后输入每行点数，自动计算坐标")
        print("\n快捷键:")
        print("  左键 - 添加点")
        print("  右键 - 删除最后一个点")
        print("  C - 清空所有点")
        print("  Enter - 计算外参")
        print("  ESC - 退出")
        print("-" * 50)

        # 加载图像
        self.load_image()

        # 创建窗口
        window_name = "Manual Extrinsic Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.update_display()

        while True:
            cv2.imshow(window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('\r') or key == ord('\n'):  # Enter键
                rvec, tvec = self.calculate_extrinsics()
                if rvec is not None:
                    save = input("\n是否保存结果? (y/n): ")
                    if save.lower() == 'y':
                        self.save_results(rvec, tvec, fov_scale)
                        break
            elif key == 27:  # ESC键
                print("\n用户取消")
                break
            elif key == ord('c'):  # 清空所有点
                self.image_points = []
                self.world_points = []
                print("清空所有点")
                self.update_display()

        cv2.destroyAllWindows()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='手动标定相机外参')
    parser.add_argument('--image', type=str, help='用于标定的图像路径')
    parser.add_argument('--fov-scale', type=float, default=0.6,
                       help='鱼眼模型视场缩放 (0.3-1.0，默认: 0.6，值越小视场越大)')

    args = parser.parse_args()

    calibrator = ManualExtrinsicCalibrator(args.image, args.fov_scale)
    calibrator.run(args.fov_scale)


if __name__ == "__main__":
    main()