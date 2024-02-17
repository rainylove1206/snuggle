# -*- coding: utf-8 -*-

from PIL import Image
import cv2 as cv
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import yaml
import textwrap
import re
import subprocess
# pip install pillow opencv-python numpy tqdm pyyaml

def normal(src_path, sy_arr, face_path=None):
    """
    “正常”图层混合模式，模拟Photoshop中的效果。
    :param src_path: 原图 path
    :param sy_arr: 水印 numpy array，其alpha通道包含了预期的不透明度
    :param face_path: 擦脸 path，可选
    :return: 新的 PIL Image 对象
    """
    # 导入原图
    arr = np.array(Image.open(src_path).convert('RGBA'), dtype=np.float32) / 255.0

    # 调整水印尺寸
    height, width, channels = arr.shape
    sy_arr = cv.resize(sy_arr, (width, height), interpolation=cv.INTER_LINEAR)

    # 导入擦脸图，将擦脸部分在水印中的对应像素设为完全透明
    if face_path is not None:
        face_arr = np.array(Image.open(face_path).convert('RGBA'), dtype=np.float32) / 255.0
        face = face_arr[..., 3] > 0.0
        sy_arr[face, 3] = 0.0

    # 找到水印中不完全透明的像素（完全透明的像素将不参与计算）
    mask = sy_arr[..., 3] > 0.0

    # RGB通道：result_rgb = rgb2 * alpha2 + rgb1 * (1 - alpha2)
    arr[mask, :3] = (sy_arr[mask, :3] * sy_arr[mask, 3, np.newaxis] +
                     arr[mask, :3] * (1 - sy_arr[mask, 3, np.newaxis]))

    # alpha通道：result_alpha = alpha2 + alpha1 * (1 - alpha2)
    arr[mask, 3] = sy_arr[mask, 3] + arr[mask, 3] * (1 - sy_arr[mask, 3])

    # 确保alpha通道的值不会超过1
    arr[..., 3] = np.clip(arr[..., 3], 0, 1)

    return Image.fromarray((arr * 255.0).astype('uint8'), 'RGBA')

def overlay(src_path, sy_arr, face_path=None):
    """
    “叠加”图层混合模式，模拟Photoshop中的效果。
    :param src_path: 原图 path
    :param sy_arr: 水印 numpy array，其alpha通道包含了预期的不透明度
    :param face_path: 擦脸 path，可选
    :return: 新的 PIL Image 对象
    """
    # 导入原图
    src_arr = np.array(Image.open(src_path).convert('RGBA'), dtype=np.float32) / 255.0

    # 调整水印尺寸
    height, width, channels = src_arr.shape
    sy_arr = cv.resize(sy_arr, (width, height), interpolation=cv.INTER_LINEAR)

    # 导入擦脸图，将擦脸部分在水印中的对应像素设为完全透明
    if face_path is not None:
        face_arr = np.array(Image.open(face_path).convert('RGBA'), dtype=np.float32) / 255.0
        face = face_arr[..., 3] > 0.0
        sy_arr[face, 3] = 0.0

    # 找出水印中不透明的像素（完全透明的像素将不参与计算）
    opaque_pixels = sy_arr[..., 3] > 0.0

    # 创建原图的副本，以保留原图的alpha和不计算部分的rgb
    arr = src_arr.copy()

    # 计算叠加模式的结果（rgb）
    mask = src_arr[opaque_pixels, :3] < 0.5
    arr[opaque_pixels, :3] = np.where(mask, 2 * src_arr[opaque_pixels, :3] * sy_arr[opaque_pixels, :3],
                                      1 - 2 * (1 - src_arr[opaque_pixels, :3]) * (1 - sy_arr[opaque_pixels, :3]))

    # 计算出的rgb乘以alpha2，然后加上rgb1乘以alpha2的补数，以考虑上层图像的透明度对最终颜色的影响
    arr[opaque_pixels, :3] = (arr[opaque_pixels, :3] * sy_arr[opaque_pixels, 3, np.newaxis]
                              + src_arr[opaque_pixels, :3] * (1 - sy_arr[opaque_pixels, 3, np.newaxis]))

    return Image.fromarray((arr * 255.0).astype('uint8'), 'RGBA')

def process_image(src_value, sy_info, sy_arr):
    sy_mode = sy_info[1]

    src_path = src_value[0]
    face_path = src_value[1] if len(src_value) > 1 else None

    # 计算图层混合
    if sy_mode in ['0', 'normal', '正常', '普白']:
        image = normal(src_path, sy_arr, face_path)
        return image
    elif sy_mode in ['1', 'overlay', '叠加', '浮雕']:
        image = overlay(src_path, sy_arr, face_path)
        return image
    else:
        return None

def save_image(image, save_format, save_path):
    try:
        if save_format == 'png':
            image.save(save_path, 'PNG', optimize=False, compress_level=6)
        elif save_format == 'jpg':
            image.convert('RGB').save(save_path, 'JPEG', quality=100, optimize=True)
        elif save_format == 'webp':
            image.save(save_path, 'WEBP', lossless=True, method=1)
    except Exception as e:
        print(f"保存图片时发生错误：{e}")

def walk_assistant(walk_path, walk_root, file_name):
    if os.path.samefile(walk_root, walk_path):
        return file_name.replace(' ', '_')
    else:
        return (os.path.relpath(walk_root, walk_path).replace(os.sep, '_') + '_' + file_name).replace(' ', '_')

def config_assistant(config_dict, item):
    config_get = config_dict.get(item, "")
    return config_get if config_get is not None else ""

def sy_info_assistant(sy_info_str, default_config):
    # 模式列表
    modes_normal = ['0', 'normal', '正常', '普白']
    modes_overlay = ['1', 'overlay', '叠加', '浮雕']

    # 分割字符串
    parts = sy_info_str.split('-')
    # 倒序匹配
    parts.reverse()

    # 提取模式
    for i, part in enumerate(parts):
        if part in modes_normal:
            mode = part
            parts.pop(i)
            break
        elif part in modes_overlay:
            mode = part
            parts.pop(i)
            break
    else:
        default_mode = str(config_assistant(default_config, '默认图层混合模式'))
        if default_mode in modes_normal + modes_overlay:
            mode = default_mode
        else:
            mode = ''

    # 检查默认不透明度
    if mode:
        default_opacity = config_assistant(default_config, '“正常”混合模式默认水印不透明度') \
            if mode in modes_normal else config_assistant(default_config, '“叠加”混合模式默认水印不透明度')
        try:
            default_opacity = float(default_opacity)
            if not 0 <= default_opacity <= 100:
                default_opacity = 200
        except ValueError:
            default_opacity = 200
    else:
        default_opacity = 200

    # 提取不透明度
    for i, part in enumerate(parts):
        if re.match(r'^\d+(\.\d+)?$', part) and 2 <= float(part) <= 100:
            opacity = float(part)
            parts.pop(i)
            break
    else:
        opacity = default_opacity

    # 剩余部分作为标签
    label = 'default_label'
    if parts:
        parts.reverse()
        label = '-'.join(parts)

    return [label, mode, opacity]

def src_info_assistant(snuggle_path):
    src_dict = {}
    for root, dirs, files in os.walk(os.path.join(snuggle_path, '1.原图')):
        for file in files:
            if '擦脸' not in file and file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file0 = walk_assistant(os.path.join(snuggle_path, '1.原图'), root, os.path.splitext(file)[0])
                src_dict[file0] = [os.path.join(root, file)]
        for file in files:
            if '擦脸' in file and file.lower().endswith('.png'):
                file0 = walk_assistant(os.path.join(snuggle_path, '1.原图'), root, os.path.splitext(file)[0].replace('擦脸', ''))
                if src_dict[file0] is not None:
                    src_dict[file0].append(os.path.join(root, file))
    return src_dict

def snuggle(snuggle_path, save_format, default_config):
    start_time = time.time()
    src_dict = src_info_assistant(snuggle_path)

    # 初始化线程池
    with ThreadPoolExecutor() as executor:
        futures = []

        for root, dirs, files in os.walk(os.path.join(snuggle_path, '2.水印')):
            for file in files:
                if not file.lower().endswith('.png'):
                    continue

                sy_info = sy_info_assistant(os.path.splitext(file)[0], default_config)
                if len(sy_info) != 3:
                    continue

                if not os.path.exists(os.path.join(snuggle_path, '3.贴膜', sy_info[0])):
                    try:
                        os.makedirs(os.path.join(snuggle_path, '3.贴膜', sy_info[0]))
                    except OSError as e:
                        print(f"创建目录失败：{e}")

                # 导入水印
                sy_arr = np.array(Image.open(os.path.join(root, file)).convert('RGBA'), dtype=np.float32) / 255.0

                # 调整水印不透明度
                if 0.0 <= float(sy_info[2]) <= 100.0:
                    sy_opacity = float(sy_info[2]) / 100.0
                else:
                    continue
                sy_arr[..., 3] *= sy_opacity

                for key, value in src_dict.items():
                    start_time_process = time.time()
                    image_processed = process_image(value, sy_info, sy_arr)
                    elapsed_time_process = time.time() - start_time_process
                    print(f"{sy_info[0]} {key} 完成 {elapsed_time_process:.3f}秒")
                    if image_processed is not None:
                        save_path = os.path.join(snuggle_path, '3.贴膜', sy_info[0], key + '.' + save_format)
                        # 立即提交保存任务到线程池
                        future = executor.submit(save_image, image_processed, save_format, save_path)
                        futures.append(future)

        # 等待所有保存操作完成
        print("等待所有图片保存完成...")
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"线程池中发生错误：{e}")

    elapsed_time = time.time() - start_time
    print(f"贴膜完成，用时{elapsed_time:.3f}秒。")

def main():
    config = {}
    if os.path.exists(os.path.join(os.getcwd(), '贴膜.yaml')):
        try:
            with open(os.path.join(os.getcwd(), '贴膜.yaml'), 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except IOError as e:
            print(f"文件读取失败：{e}")
    else:
        try:
            with open(os.path.join(os.getcwd(), '贴膜.yaml'), 'w', encoding='utf-8') as file:
                content = textwrap.dedent("""
                                贴膜路径: 
                                出图格式: 
                                
                                默认图层混合模式: 
                                “正常”混合模式默认水印不透明度: 
                                “叠加”混合模式默认水印不透明度: 
                                
                                自动打开文件夹: 
                                检查水印透明度通道最大值: 
    
                                ############################################################
                                
                                # 说明：
                                # 出图格式选填 png/jpg/webp 。
                                # 图层混合模式选填0或1，0为“正常”，1为“叠加”。直接写中文字也可以。
                                # 水印不透明度选填0-100之间的数，0为完全透明，100为完全不透明。
                                # 剩余功能项，如果需要，可填true开启对应功能，不填或填false为关闭。
                                
                                # 贴膜参数由水印的文件名指定。水印命名格式为：标签-模式-不透明度.png
                                # 标签可以是贴膜QQ号或者其他一些用于区分的内容（没有其他作用）。
                                # 模式选填0或1，0为“正常”，1为“叠加”。不透明度为2-100之间的数。
                                # 如果已经在本yaml文件中填写了默认值，可以省略。示例：
                                # 10100-0-20.png （“正常”模式，20%不透明度）
                                # 10200-1-60.png （“叠加”模式，60不透明度）
                                # 20200-100.png  （100%不透明度，图层混合模式按本文件中填写的默认值，没填或填错这个水印就不贴）
                                # 30020-0100.png （图层混合模式和不透明度都按本文件中填写的默认值，没填或填错这个水印就不贴）
                            """).strip()
                file.write(content)
        except IOError as e:
            print(f"文件写入失败：{e}")

    try:
        if config.get("贴膜路径", "") is not None and os.path.exists(config.get("贴膜路径", "")):
            PATH = config["贴膜路径"]
            print("贴膜路径：", PATH)
        else:
            while True:
                PATH = input("贴膜路径：")
                if os.path.exists(PATH):
                    break
                else:
                    print("路径不存在，请重新输入。")

        if config.get("出图格式", "") is not None and config.get("出图格式", "") in ['png', 'jpg', 'webp']:
            SAVEFORMAT = config["出图格式"]
            print("出图格式：", SAVEFORMAT)
        else:
            while True:
                SAVEFORMAT = input("出图格式：（png/jpg/webp）")
                if SAVEFORMAT in ['png', 'jpg', 'webp']:
                    break
                else:
                    print("只能是png/jpg/webp格式哦~")

        print('')
    except:
        return

    for dir in ['1.原图', '2.水印', '3.贴膜']:
        if not os.path.exists(os.path.join(PATH, dir)):
            try:
                os.makedirs(os.path.join(PATH, dir))
            except OSError as e:
                print(f"创建目录失败：{e}")

        # 自动打开这三个文件夹（可选）（仅Windows平台）
        if config_assistant(config, '自动打开文件夹'):
            subprocess.run(f'explorer {os.path.join(PATH, dir)}')

    while True:
        try:
            input("按Enter键开始贴膜~")
            snuggle(PATH, SAVEFORMAT, config)
            print('')
        except:
            break

if __name__ == "__main__":
    main()


