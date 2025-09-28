#!/usr/bin/env python
"""
相机内参标定主程序
使用张正友标定法对相机进行标定
"""

import cv2
import numpy as np
import yaml
import glob
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime

# 获取项目根目录（脚本所在目录的父目录的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 文件路径和名称常量
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "calibration_images"
DEFAULT_IMAGE_PATTERN = str(PROJECT_ROOT / "calibration_images" / "*.png")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "internal_paras"
OUTPUT_YAML_FILENAME = "calibration.yaml"


class CameraCalibrator:
    def __init__(self, config_path: str = None):
        """初始化标定器

        Args:
            config_path (str, optional): 配置文件路径. Defaults to None.
                如果为None，使用默认路径 'config/config.yaml'
        """
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config" / "internal_config.yaml")
        self.config = self._load_config(config_path)
        self.checkerboard_size = (
            self.config['checkerboard']['cols'],
            self.config['checkerboard']['rows']
        )
        self.square_size = self.config['checkerboard']['square_size']

        # 准备棋盘格世界坐标点
        self.objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # 存储标定数据
        self.objpoints = []  # 世界坐标系中的点
        self.imgpoints = []  # 图像坐标系中的点
        self.image_names = []  # 使用的图像名称
        self.image_errors = []  # 每张图像的重投影误差

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件

        Args:
            config_path (str): YAML配置文件路径

        Returns:
            dict: 配置参数字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def find_corners(self, img_path: str) -> Tuple[bool, np.ndarray, np.ndarray]:
        """检测棋盘格角点

        Args:
            img_path (str): 图像文件的完整路径

        Returns:
            tuple: 包含三个元素的元组
                - bool: 是否成功检测到棋盘格
                - np.ndarray: 角点坐标数组，shape为(n, 1, 2)，失败时为None
                - np.ndarray: 绘制了角点的图像，失败时为原图
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return False, None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)

        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                       self.config['calibration']['subpix_criteria']['max_iter'],
                       self.config['calibration']['subpix_criteria']['epsilon'])
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 绘制角点
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, self.checkerboard_size, corners, ret)

            return True, corners, img_with_corners

        return False, None, img

    def collect_calibration_images(self, image_pattern: str = DEFAULT_IMAGE_PATTERN) -> int:
        """收集并处理标定图像

        Args:
            image_pattern (str, optional): 图像文件的glob匹配模式.
                Defaults to DEFAULT_IMAGE_PATTERN.

        Returns:
            int: 成功检测到棋盘格的图像数量
        """
        image_paths = sorted(glob.glob(image_pattern))

        if not image_paths:
            print(f"未找到匹配的图像: {image_pattern}")
            return 0

        print(f"找到 {len(image_paths)} 张图像")
        successful_images = 0

        for img_path in image_paths:
            print(f"处理图像: {Path(img_path).name}", end=" ... ")

            ret, corners, _ = self.find_corners(img_path)

            if ret:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                self.image_names.append(Path(img_path).name)
                successful_images += 1
                print("成功")
            else:
                print("失败 (未检测到棋盘格)")

        print(f"\n成功处理 {successful_images}/{len(image_paths)} 张图像")
        return successful_images

    def calibrate(self) -> Tuple[bool, Dict]:
        """执行相机标定

        Returns:
            tuple: 包含两个元素的元组
                - bool: 标定是否成功
                - dict: 标定结果字典，包含以下键值:
                    - calibration_date: 标定日期
                    - image_width: 图像宽度
                    - image_height: 图像高度
                    - camera_matrix: 相机内参矩阵
                    - distortion_coefficients: 畸变系数
                    - rms_error: RMS重投影误差
                    - num_images_used: 使用的图像数量
        """
        if len(self.objpoints) < self.config['calibration']['min_images']:
            print(f"图像数量不足: {len(self.objpoints)} < {self.config['calibration']['min_images']}")
            return False, {}

        # 获取图像尺寸
        sample_img = cv2.imread(glob.glob(str(DEFAULT_IMAGE_DIR / "*"))[0])
        h, w = sample_img.shape[:2]

        # 获取模型类型
        model = self.config['calibration'].get('model', 'pinhole')
        print(f"\n开始标定 ({'鱼眼' if model == 'fisheye' else '标准'}模型)...")

        if model == 'fisheye':
            # 准备鱼眼标定参数
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))

            # 转换数据格式为鱼眼模型需要的格式
            objpoints = [p.reshape(1, -1, 3) for p in self.objpoints]
            imgpoints = [p.reshape(1, -1, 2) for p in self.imgpoints]

            # 鱼眼标定标志
            calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

            # 执行鱼眼标定
            try:
                ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objpoints, imgpoints, (w, h),
                    K, D, flags=calibration_flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
            except cv2.error as e:
                print(f"标定失败: {e}")
                return False, {}
        else:
            # 标准针孔模型标定
            ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, (w, h),
                None, None, flags=self.config['calibration']['flags']
            )

        if not ret:
            print("标定失败")
            return False, {}

        # 计算重投影误差
        total_error = 0
        errors = []

        if model == 'fisheye':
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.fisheye.projectPoints(
                    objpoints[i].reshape(-1, 1, 3), rvecs[i], tvecs[i], K, D
                )
                error = cv2.norm(imgpoints[i].reshape(-1, 1, 2), imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                errors.append(error)
                total_error += error
        else:
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], K, D)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                errors.append(error)
                total_error += error

        mean_error = total_error / (len(objpoints) if model == 'fisheye' else len(self.objpoints))

        # 过滤高误差图像
        max_error = self.config['calibration']['max_error']
        good_indices = [i for i, e in enumerate(errors) if e < max_error]

        if len(good_indices) < len(errors):
            print(f"\n重新标定 (剔除 {len(errors) - len(good_indices)} 张高误差图像)...")

            if model == 'fisheye':
                good_objpoints = [objpoints[i] for i in good_indices]
                good_imgpoints = [imgpoints[i] for i in good_indices]

                try:
                    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                        good_objpoints, good_imgpoints, (w, h),
                        K, D, flags=calibration_flags,
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
                except cv2.error as e:
                    print(f"重新标定失败: {e}")
                    return False, {}

                # 重新计算误差
                total_error = 0
                errors = []
                for i in range(len(good_objpoints)):
                    imgpoints2, _ = cv2.fisheye.projectPoints(
                        good_objpoints[i].reshape(-1, 1, 3), rvecs[i], tvecs[i], K, D
                    )
                    error = cv2.norm(good_imgpoints[i].reshape(-1, 1, 2), imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    errors.append(error)
                    total_error += error
                mean_error = total_error / len(good_objpoints)
            else:
                good_objpoints = [self.objpoints[i] for i in good_indices]
                good_imgpoints = [self.imgpoints[i] for i in good_indices]

                ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                    good_objpoints, good_imgpoints, (w, h),
                    None, None, flags=self.config['calibration']['flags']
                )

                # 重新计算误差
                total_error = 0
                errors = []
                for i in range(len(good_objpoints)):
                    imgpoints2, _ = cv2.projectPoints(good_objpoints[i], rvecs[i], tvecs[i], K, D)
                    error = cv2.norm(good_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    errors.append(error)
                    total_error += error
                mean_error = total_error / len(good_objpoints)

            self.image_errors = [(self.image_names[good_indices[i]], errors[i]) for i in range(len(errors))]
        else:
            self.image_errors = [(self.image_names[i], errors[i]) for i in range(len(errors))]

        print(f"标定完成 ({'鱼眼' if model == 'fisheye' else '标准'}模型)!")
        print(f"RMS重投影误差: {mean_error:.4f} 像素")
        print(f"使用图像数量: {len(good_indices) if len(good_indices) < len(errors) else len(errors)}")

        # 构建结果字典
        calibration_result = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model,  # 标记模型类型
            'image_width': w,
            'image_height': h,
            'camera_matrix': K.tolist(),
            'distortion_coefficients': D.flatten().tolist(),  # k1, k2, k3, k4
            'calibration_flags': int(calibration_flags),
            'rms_error': float(mean_error),
            'num_images_used': len(good_indices) if len(good_indices) < len(errors) else len(errors),
            'image_errors': [{'image': name, 'error': float(error)} for name, error in self.image_errors]
        }

        self.calibration_result = calibration_result
        self.camera_matrix = K
        self.distortion_coeffs = D

        return True, calibration_result

    def save_results(self, output_dir: str = None):
        """保存标定结果到YAML文件

        Args:
            output_dir (str, optional): 输出目录路径. Defaults to None.
                如果为None，使用默认路径
        """
        if output_dir is None:
            output_dir = str(DEFAULT_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)

        # 保存YAML格式 (ROS/OpenCV标准格式)
        yaml_path = os.path.join(output_dir, OUTPUT_YAML_FILENAME)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            # 格式化相机矩阵和畸变系数为标准OpenCV格式
            yaml_data = {
                'calibration_date': self.calibration_result['calibration_date'],
                'image_width': self.calibration_result['image_width'],
                'image_height': self.calibration_result['image_height'],
                'camera_matrix': {
                    'rows': 3,
                    'cols': 3,
                    'data': self.calibration_result['camera_matrix']
                },
                'model': self.calibration_result['model'],
                'distortion_coefficients': {
                    'rows': 1,
                    'cols': len(self.calibration_result['distortion_coefficients']),
                    'data': self.calibration_result['distortion_coefficients']
                },
                'calibration_flags': self.calibration_result['calibration_flags'],
                'rms_error': self.calibration_result['rms_error'],
                'num_images_used': self.calibration_result['num_images_used'],
                'image_errors': self.calibration_result['image_errors']
            }
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        print(f"标定结果已保存至: {yaml_path}")


def main():
    """主函数"""
    print("=" * 50)
    print("相机内参标定程序")
    print("=" * 50)

    # 创建标定器
    calibrator = CameraCalibrator()  # 使用默认配置路径

    # 收集标定图像
    num_images = calibrator.collect_calibration_images()

    if num_images == 0:
        print("错误: 没有可用的标定图像")
        return

    # 执行标定
    success, result = calibrator.calibrate()

    if success:
        # 保存结果
        print("\n保存标定结果...")
        calibrator.save_results()

        print("\n标定完成!")
        print("运行 'python scripts/evaluate.py' 查看详细评估报告")
    else:
        print("标定失败")


if __name__ == "__main__":
    main()