#!/usr/bin/env python
"""
标定质量评估脚本
从calibrate.py中提取的评估功能
"""

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 文件名常量
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "internal_paras"
DEFAULT_CALIBRATION_PATH = DEFAULT_OUTPUT_DIR / "calibration.yaml"
OUTPUT_ERROR_REPORT_FILENAME = "error_report.txt"
OUTPUT_ERROR_PLOT_FILENAME = "error_analysis.png"


def load_config(config_path: str) -> dict:
    """加载配置文件

    Args:
        config_path (str): YAML配置文件路径

    Returns:
        dict: 配置参数字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_calibration_result(result_path: str) -> dict:
    """加载标定结果

    Args:
        result_path (str): 标定结果YAML文件路径

    Returns:
        dict: 标定结果字典
    """
    with open(result_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_error_report(output_dir: str, calibration_result: dict, image_errors: list, config: dict):
    """生成并保存误差分析报告

    Args:
        output_dir (str): 输出目录路径
        calibration_result (dict): 标定结果字典
        image_errors (list): 图像误差列表，格式为[(图像名, 误差值), ...]
        config (dict): 配置参数字典
    """
    report_path = os.path.join(output_dir, OUTPUT_ERROR_REPORT_FILENAME)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("相机标定误差报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"标定日期: {calibration_result['calibration_date']}\n")
        f.write(f"图像分辨率: {calibration_result['image_width']}x{calibration_result['image_height']}\n")
        f.write(f"使用图像数: {calibration_result['num_images_used']}\n")
        f.write(f"平均RMS误差: {calibration_result['rms_error']:.4f} 像素\n\n")

        f.write("各图像重投影误差:\n")
        f.write("-" * 30 + "\n")

        errors_sorted = sorted(image_errors, key=lambda x: x[1])
        for img_name, error in errors_sorted:
            status = "✓" if error < config['calibration']['max_error'] else "✗"
            f.write(f"{status} {img_name:30s} {error:.4f} px\n")

        f.write("\n统计信息:\n")
        f.write("-" * 30 + "\n")
        errors_only = [e for _, e in image_errors]
        f.write(f"最小误差: {min(errors_only):.4f} px\n")
        f.write(f"最大误差: {max(errors_only):.4f} px\n")
        f.write(f"误差标准差: {np.std(errors_only):.4f} px\n")

    print(f"误差报告已保存至: {report_path}")


def plot_errors(output_dir: str, image_errors: list, config: dict):
    """绘制误差分布图

    Args:
        output_dir (str): 输出目录路径
        image_errors (list): 图像误差列表，格式为[(图像名, 误差值), ...]
        config (dict): 配置参数字典
    """
    if not image_errors:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 误差条形图
    names = [name[:15] + "..." if len(name) > 15 else name for name, _ in image_errors]
    errors = [error for _, error in image_errors]

    axes[0].bar(range(len(errors)), errors)
    axes[0].axhline(y=config['calibration']['max_error'], color='r', linestyle='--',
                    label=f"Threshold ({config['calibration']['max_error']} px)")
    axes[0].set_xlabel('Image Index')
    axes[0].set_ylabel('Reprojection Error (pixels)')
    axes[0].set_title('Reprojection Error by Image')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 误差直方图
    axes[1].hist(errors, bins=15, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=config['calibration']['max_error'], color='r', linestyle='--',
                    label=f"Threshold ({config['calibration']['max_error']} px)")
    axes[1].set_xlabel('Reprojection Error (pixels)')
    axes[1].set_ylabel('Number of Images')
    axes[1].set_title('Error Distribution Histogram')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, OUTPUT_ERROR_PLOT_FILENAME)
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"误差分析图已保存至: {plot_path}")


def print_calibration_results(calibration_result: dict):
    """打印标定结果到控制台

    Args:
        calibration_result (dict): 标定结果字典，包含camera_matrix、
            distortion_coefficients、rms_error等键值
    """
    print("\n" + "=" * 50)
    model_type = calibration_result.get('model', 'pinhole')
    model_name = "鱼眼模型" if model_type == 'fisheye' else "针孔模型"
    print(f"相机内参标定结果 ({model_name})")
    print("=" * 50)

    print("\n相机内参矩阵 K:")
    print("-" * 30)
    K = np.array(calibration_result['camera_matrix']['data']).reshape(3, 3)
    print(f"fx = {K[0,0]:.2f} (焦距x)")
    print(f"fy = {K[1,1]:.2f} (焦距y)")
    print(f"cx = {K[0,2]:.2f} (主点x)")
    print(f"cy = {K[1,2]:.2f} (主点y)")

    print("\n畸变系数 D:")
    print("-" * 30)
    D = calibration_result['distortion_coefficients']['data']
    model_type = calibration_result.get('model', 'pinhole')

    if model_type == 'fisheye':
        # 鱼眼模型: k1, k2, k3, k4
        print(f"k1 = {D[0]:.6f} (径向畸变1)")
        if len(D) > 1:
            print(f"k2 = {D[1]:.6f} (径向畸变2)")
        if len(D) > 2:
            print(f"k3 = {D[2]:.6f} (径向畸变3)")
        if len(D) > 3:
            print(f"k4 = {D[3]:.6f} (径向畸变4)")
    else:
        # 标准模型: k1, k2, p1, p2, [k3]
        print(f"k1 = {D[0]:.6f} (径向畸变1)")
        print(f"k2 = {D[1]:.6f} (径向畸变2)")
        print(f"p1 = {D[2]:.6f} (切向畸变1)")
        print(f"p2 = {D[3]:.6f} (切向畸变2)")
        if len(D) > 4:
            print(f"k3 = {D[4]:.6f} (径向畸变3)")

    print("\n标定质量:")
    print("-" * 30)
    print(f"RMS重投影误差: {calibration_result['rms_error']:.4f} 像素")

    if calibration_result['rms_error'] < 0.5:
        print("✓ 优秀 (< 0.5 px)")
    elif calibration_result['rms_error'] < 1.0:
        print("✓ 良好 (< 1.0 px)")
    else:
        print("⚠ 需要改进 (> 1.0 px)")


def main():
    """主函数

    加载标定结果并打印评估报告
    """
    # 加载配置和标定结果
    config_path = str(PROJECT_ROOT / "config" / "config.yaml")
    result_path = str(DEFAULT_CALIBRATION_PATH)
    output_dir = str(DEFAULT_OUTPUT_DIR)

    try:
        config = load_config(config_path)
        calibration_result = load_calibration_result(result_path)

        # 打印标定结果
        print_calibration_results(calibration_result)

        # 检查并处理image_errors数据
        if 'image_errors' in calibration_result:
            # 转换格式：从字典列表转为元组列表
            image_errors = [(item['image'], item['error']) for item in calibration_result['image_errors']]

            # 生成误差报告
            save_error_report(output_dir, calibration_result, image_errors, config)

            # 绘制误差分布图
            plot_errors(output_dir, image_errors, config)
        else:
            print("\n注意: 标定结果中没有详细的图像误差数据")
            print("请重新运行标定程序以生成完整的误差分析")

    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()