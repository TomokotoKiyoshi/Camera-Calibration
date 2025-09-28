#!/usr/bin/env python
"""
标定验证脚本
使用标定结果对图像进行去畸变处理，验证标定效果
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path
import argparse

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认路径
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "internal_paras"
DEFAULT_CALIBRATION_PATH = DEFAULT_OUTPUT_DIR / "calibration.yaml"
DEFAULT_VERIFICATION_DIR = DEFAULT_OUTPUT_DIR / "verification"
DEFAULT_TEST_IMAGE_PATTERN = str(PROJECT_ROOT / "calibration_images" / "*.png")


class CalibrationVerifier:
    def __init__(self, calibration_path: str = None, fov_scale: float = 0.6):
        """初始化验证器

        Args:
            calibration_path (str, optional): 标定结果文件路径. Defaults to None.
            fov_scale (float, optional): 鱼眼模型的视场缩放因子 (0-1). Defaults to 0.6.
        """
        if calibration_path is None:
            calibration_path = str(DEFAULT_CALIBRATION_PATH)

        self.calibration_data = self._load_calibration(calibration_path)
        self._parse_calibration_data(fov_scale)

    def _load_calibration(self, filepath: str) -> dict:
        """加载标定结果

        Args:
            filepath (str): 标定结果YAML文件路径

        Returns:
            dict: 标定结果字典

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"标定文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _parse_calibration_data(self, fov_scale=0.6):
        """解析标定数据

        Args:
            fov_scale (float): 鱼眼模型的视场缩放因子 (0-1)，值越小视场越大
        """
        # 检查模型类型
        self.model = self.calibration_data.get('model', 'pinhole')

        # 相机矩阵
        if isinstance(self.calibration_data['camera_matrix'], dict):
            self.camera_matrix = np.array(
                self.calibration_data['camera_matrix']['data']
            ).reshape(3, 3)
        else:
            self.camera_matrix = np.array(self.calibration_data['camera_matrix'])

        # 畸变系数
        if isinstance(self.calibration_data['distortion_coefficients'], dict):
            self.dist_coeffs = np.array(
                self.calibration_data['distortion_coefficients']['data']
            )
        else:
            self.dist_coeffs = np.array(
                self.calibration_data['distortion_coefficients']
            )

        # 图像尺寸
        self.image_width = self.calibration_data['image_width']
        self.image_height = self.calibration_data['image_height']

        # 根据模型类型计算新的相机矩阵
        if self.model == 'fisheye':
            # 鱼眼模型：缩放相机矩阵以调整视场
            self.new_camera_matrix = self.camera_matrix.copy()
            # 缩放焦距以增大视场（值越小，视场越大）
            self.new_camera_matrix[0, 0] *= fov_scale  # fx
            self.new_camera_matrix[1, 1] *= fov_scale  # fy
            # 保持主点在图像中心
            self.new_camera_matrix[0, 2] = self.image_width / 2   # cx
            self.new_camera_matrix[1, 2] = self.image_height / 2  # cy
            self.roi = (0, 0, self.image_width, self.image_height)  # 鱼眼模型不裁剪
        else:
            # 标准模型
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs,
                (self.image_width, self.image_height), 1,
                (self.image_width, self.image_height)
            )

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """对图像进行去畸变

        Args:
            img (np.ndarray): 输入图像

        Returns:
            np.ndarray: 去畸变后的图像
        """
        if img is None:
            return None

        # 根据模型类型选择去畸变方法
        if self.model == 'fisheye':
            # 鱼眼模型去畸变
            undistorted = cv2.fisheye.undistortImage(
                img, self.camera_matrix, self.dist_coeffs.reshape(-1, 1),
                None, self.new_camera_matrix
            )
        else:
            # 标准模型去畸变
            undistorted = cv2.undistort(
                img, self.camera_matrix, self.dist_coeffs,
                None, self.new_camera_matrix
            )

            # 裁剪ROI（仅标准模型）
            x, y, w, h = self.roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def verify_with_live_camera(self, device_id: int = 0):
        """使用实时摄像头验证标定效果

        Args:
            device_id (int, optional): 摄像头设备ID. Defaults to 0.
        """
        print("=" * 50)
        print("实时相机去畸变验证")
        print("=" * 50)
        print("按 'q' 退出")
        print("按 's' 保存当前帧")
        print("按 'space' 暂停/继续")
        print("-" * 50)

        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {device_id}")
            return

        # 设置摄像头分辨率为标定时的分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

        paused = False
        save_count = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("错误: 无法读取摄像头图像")
                    break

                # 根据模型类型去畸变
                if self.model == 'fisheye':
                    undistorted = cv2.fisheye.undistortImage(
                        frame, self.camera_matrix, self.dist_coeffs.reshape(-1, 1),
                        None, self.new_camera_matrix
                    )
                else:
                    undistorted = cv2.undistort(
                        frame, self.camera_matrix, self.dist_coeffs,
                        None, self.new_camera_matrix
                    )

                # 只显示去畸变图像 - 固定1920x1080
                target_width = 1920
                target_height = 1080

                undistorted_resized = cv2.resize(undistorted, (target_width, target_height))

                # 添加标签
                cv2.putText(undistorted_resized, "Undistorted", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                if paused:
                    cv2.putText(undistorted_resized, "PAUSED", (target_width - 150, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # 显示
                cv2.namedWindow("Live Camera Verification", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Live Camera Verification", target_width, target_height)
                cv2.imshow("Live Camera Verification", undistorted_resized)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print("已暂停")
                else:
                    print("已继续")
            elif key == ord('s') and not paused:
                # 保存当前帧
                save_dir = DEFAULT_VERIFICATION_DIR
                save_dir.mkdir(exist_ok=True)

                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = save_dir / f"verified_{timestamp}_{save_count:03d}.png"
                cv2.imwrite(str(save_path), undistorted_resized)
                save_count += 1
                print(f"已保存: {save_path}")

        cap.release()
        cv2.destroyAllWindows()

    def batch_verify(self, save_results: bool = False):
        """批量验证图像

        Args:
            save_results (bool, optional): 是否保存结果. Defaults to False.
        """
        import glob
        image_paths = sorted(glob.glob(DEFAULT_TEST_IMAGE_PATTERN))

        if not image_paths:
            print(f"未找到匹配的图像: {DEFAULT_TEST_IMAGE_PATTERN}")
            return

        print(f"找到 {len(image_paths)} 张图像")

        if save_results:
            output_dir = DEFAULT_VERIFICATION_DIR
            output_dir.mkdir(exist_ok=True)
            print(f"结果将保存到: {output_dir}")

        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] 处理: {Path(img_path).name}")

            # 读取原图
            original = cv2.imread(img_path)
            if original is None:
                print(f"  无法读取图像: {img_path}")
                continue

            # 获取去畸变图像
            undistorted = self.undistort_image(original)

            if save_results and undistorted is not None:
                # 创建并保存对比图
                h, w = original.shape[:2]
                comparison = np.hstack([original, undistorted])
                comparison_path = output_dir / f"comparison_{Path(img_path).name}"
                cv2.imwrite(str(comparison_path), comparison)
                print(f"  对比图已保存: {comparison_path.name}")

            if not save_results:
                # 显示对比图
                h, w = original.shape[:2]
                display_width = 800
                scale = display_width / w
                display_height = int(h * scale)

                original_resized = cv2.resize(original, (display_width, display_height))
                undistorted_resized = cv2.resize(undistorted, (display_width, display_height))
                comparison_display = np.hstack([original_resized, undistorted_resized])

                cv2.putText(comparison_display, "Original", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison_display, "Undistorted", (display_width + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Distortion Correction", comparison_display)
                print(f"  显示: {Path(img_path).name} (按任意键继续，'q'退出)")

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break

        cv2.destroyAllWindows()
        print("\n验证完成")



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证相机标定结果')
    parser.add_argument('--calibration', type=str,
                       help='标定结果文件路径')
    parser.add_argument('--live', action='store_true',
                       help='使用实时摄像头验证')
    parser.add_argument('--device', type=int, default=1,
                       help='摄像头设备ID (默认: 1)')
    parser.add_argument('--save', action='store_true',
                       help='保存去畸变结果')
    parser.add_argument('--fov-scale', type=float, default=0.6,
                       help='鱼眼模型视场缩放 (0.3-1.0，默认: 0.6，值越小视场越大)')

    args = parser.parse_args()

    try:
        # 创建验证器
        verifier = CalibrationVerifier(args.calibration, args.fov_scale)

        # 根据参数执行不同的验证模式
        if args.live:
            # 实时摄像头验证
            verifier.verify_with_live_camera(args.device)
        elif args.save:
            # 批量保存对比图
            verifier.batch_verify(save_results=True)
        else:
            # 默认：批量验证
            print("\n开始批量验证...")
            verifier.batch_verify()

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行标定程序生成标定结果")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()