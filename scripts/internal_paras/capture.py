#!/usr/bin/env python
"""
相机标定图像采集程序
实时检测棋盘格并采集标定图像
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from datetime import datetime
import time

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "calibration_images"


class CalibrationCapture:
    def __init__(self, config_path: str = None):
        """初始化采集器

        Args:
            config_path (str, optional): 配置文件路径. Defaults to None.
        """
        if config_path is None:
            config_path = str(DEFAULT_CONFIG_PATH)

        self.config = self._load_config(config_path)
        self.checkerboard_size = (
            self.config['checkerboard']['cols'],
            self.config['checkerboard']['rows']
        )

        # 采集参数
        self.device_id = self.config['capture']['device_id']
        self.target_images = self.config['capture']['target_images']
        self.save_format = self.config['capture']['save_format']
        self.preview_size = tuple(self.config['capture']['preview_size'])

        # 创建保存目录
        self.save_dir = str(DEFAULT_IMAGE_DIR)
        os.makedirs(self.save_dir, exist_ok=True)

        # 统计信息
        self.captured_count = 0
        self.last_capture_time = 0
        self.capture_delay = 2  # 两次采集之间的最小间隔（秒）

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件

        Args:
            config_path (str): YAML配置文件路径

        Returns:
            dict: 配置参数字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _check_sharpness(self, gray_img: np.ndarray) -> float:
        """检查图像清晰度

        Args:
            gray_img (np.ndarray): 灰度图像

        Returns:
            float: 清晰度分数（Laplacian方差）
        """
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        return laplacian.var()

    def _suggest_next_pose(self) -> str:
        """建议下一个采集姿态

        Returns:
            str: 姿态建议
        """
        suggestions = [
            "Face board directly",
            "Tilt left ~30 deg",
            "Tilt right ~30 deg",
            "Tilt up ~30 deg",
            "Tilt down ~30 deg",
            "Tilt upper-left",
            "Tilt upper-right",
            "Tilt lower-left",
            "Tilt lower-right",
            "Move closer",
            "Move farther",
            "Rotate ~45 deg"
        ]

        if self.captured_count < len(suggestions):
            return suggestions[self.captured_count % len(suggestions)]
        return "Any angle"

    def detect_checkerboard(self, frame: np.ndarray) -> tuple:
        """检测棋盘格

        Args:
            frame (np.ndarray): 输入图像

        Returns:
            tuple: (是否检测到, 角点, 处理后的图像, 清晰度, 覆盖率, 质量是否良好)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)

        # 检查清晰度
        sharpness = self._check_sharpness(gray)

        # 绘制结果
        display_frame = frame.copy()

        if ret:
            # 亚像素优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 绘制角点
            cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners, ret)

            # 计算棋盘格在图像中的覆盖率
            x_coords = corners[:, 0, 0]
            y_coords = corners[:, 0, 1]
            coverage_x = (x_coords.max() - x_coords.min()) / frame.shape[1]
            coverage_y = (y_coords.max() - y_coords.min()) / frame.shape[0]
            coverage = min(coverage_x, coverage_y)

            # 显示质量信息
            quality_color = (0, 255, 0) if sharpness > 100 else (0, 165, 255)
            cv2.putText(display_frame, f"Sharpness: {sharpness:.0f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
            cv2.putText(display_frame, f"Coverage: {coverage*100:.0f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

            # 质量判断
            good_quality = sharpness > 100 and coverage > 0.3 and coverage < 0.8

            return True, corners, display_frame, sharpness, coverage, good_quality
        else:
            # 未检测到棋盘格
            cv2.putText(display_frame, "No Checkerboard Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            return False, None, display_frame, sharpness, 0, False

    def save_image(self, frame: np.ndarray) -> str:
        """保存图像

        Args:
            frame (np.ndarray): 要保存的图像

        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calib_{timestamp}_{self.captured_count:03d}.{self.save_format}"
        filepath = os.path.join(self.save_dir, filename)

        cv2.imwrite(filepath, frame)
        return filepath

    def run(self):
        """运行采集程序"""
        print("=" * 50)
        print("相机标定图像采集程序")
        print("=" * 50)
        print(f"目标采集数量: {self.target_images}")
        print(f"棋盘格规格: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} 内角点")
        print(f"保存格式: {self.save_format}")
        print(f"保存目录: {self.save_dir}")
        print("\n操作说明:")
        print("  空格键 - 手动采集图像")
        print("  'a' 键 - 自动采集模式")
        print("  'q' 键 - 退出采集")
        print("  's' 键 - 显示/隐藏建议")
        print("-" * 50)

        # 打开摄像头
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.device_id}")
            return

        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_size[1])

        auto_capture = False
        show_suggestion = True

        print("\n开始采集...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头图像")
                break

            # 检测棋盘格
            detected, corners, display_frame, sharpness, coverage, good_quality = self.detect_checkerboard(frame)

            # 显示状态信息
            status_color = (0, 255, 0) if detected and good_quality else (0, 165, 255) if detected else (0, 0, 255)
            status_text = "Ready" if detected and good_quality else "Low Quality" if detected else "No Board"

            cv2.putText(display_frame, f"Status: {status_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"Captured: {self.captured_count}/{self.target_images}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示建议
            if show_suggestion:
                suggestion = self._suggest_next_pose()
                cv2.putText(display_frame, f"Suggestion: {suggestion}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 自动采集模式
            if auto_capture and detected and good_quality:
                current_time = time.time()
                if current_time - self.last_capture_time > self.capture_delay:
                    filepath = self.save_image(frame)
                    self.captured_count += 1
                    self.last_capture_time = current_time
                    print(f"[{self.captured_count}/{self.target_images}] 自动保存: {Path(filepath).name} "
                          f"(清晰度: {sharpness:.0f}, 覆盖率: {coverage*100:.0f}%)")

                    if self.captured_count >= self.target_images:
                        print("\n✓ 已达到目标数量!")
                        break

            # 显示模式指示
            if auto_capture:
                cv2.putText(display_frame, "AUTO MODE", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示图像
            cv2.imshow("Calibration Capture", display_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # 空格键 - 手动采集
                if detected:
                    filepath = self.save_image(frame)
                    self.captured_count += 1
                    print(f"[{self.captured_count}/{self.target_images}] 手动保存: {Path(filepath).name} "
                          f"(清晰度: {sharpness:.0f}, 覆盖率: {coverage*100:.0f}%)")

                    if not good_quality:
                        print("  ⚠ 警告: 图像质量较低")

                    if self.captured_count >= self.target_images:
                        print("\n✓ 已达到目标数量!")
                        break
                else:
                    print("未检测到棋盘格，无法保存")

            elif key == ord('a'):  # 切换自动采集
                auto_capture = not auto_capture
                mode = "开启" if auto_capture else "关闭"
                print(f"自动采集模式: {mode}")

            elif key == ord('s'):  # 切换建议显示
                show_suggestion = not show_suggestion

            elif key == ord('q'):  # 退出
                print("\n用户取消采集")
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

        # 打印统计
        print("\n" + "=" * 50)
        print("采集完成")
        print(f"共采集图像: {self.captured_count} 张")
        print(f"保存位置: {self.save_dir}/")

        if self.captured_count < self.config['calibration']['min_images']:
            print(f"\n⚠ 警告: 采集数量不足")
            print(f"  最少需要: {self.config['calibration']['min_images']} 张")
            print(f"  当前只有: {self.captured_count} 张")
        else:
            print("\n✓ 可以运行标定程序了")
            print("  运行: python scripts/calibrate.py")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='采集相机标定图像')
    parser.add_argument('--device', type=int, default=0, help='摄像头设备ID')

    args = parser.parse_args()

    # 创建采集器（使用默认配置）
    capture = CalibrationCapture()

    # 如果指定了设备ID，覆盖配置
    if args.device is not None:
        capture.device_id = args.device

    # 运行采集
    capture.run()


if __name__ == "__main__":
    main()