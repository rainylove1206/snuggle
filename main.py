from PIL import Image
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
import time
import yaml
import textwrap
import re
# pip install pillow opencv-python numpy tqdm pyyaml

def normal(image1, image2, image3=None):
    """
    “正常”图层混合模式
    :param image1: 原图numpy数组
    :param image2: 水印numpy数组
    :param image3: 擦脸numpy数组
    :return: 新的numpy数组
    """
    # 将擦脸部分在水印中的对应像素设为完全透明（image2会被修改）
    if image3 is not None:
        face = image3[..., 3] > 0.0
        image2[face, 3] = 0.0

    # 将arr初始化为原图的副本（image1不会被修改）
    arr = image1.copy()

    # 找到水印中不完全透明的像素（完全透明的像素将不参与计算）
    mask = image2[..., 3] > 0.0

    # RGB通道：result_rgb = image2_rgb * alpha2 + image1_rgb * (1 - alpha2) * alpha1
    arr[..., :3][mask] = (image2[..., :3][mask] * image2[..., 3, np.newaxis][mask] + image1[..., :3][mask]
                          * (1 - image2[..., 3, np.newaxis][mask]) * image1[..., 3, np.newaxis][mask])

    # alpha通道：result_alpha = alpha2 + (1 - alpha2) * alpha1
    arr[..., 3][mask] = image2[..., 3][mask] + image1[..., 3][mask] * (1 - image2[..., 3][mask])

    return arr

def overlay(image1, image2, image3=None):
    """
    “叠加”图层混合模式
    :param image1: 原图numpy数组
    :param image2: 水印numpy数组
    :param image3: 擦脸numpy数组
    :return: 新的numpy数组
    """
    # 将擦脸部分在水印中的对应像素设为完全透明（image2会被修改）
    if image3 is not None:
        face = image3[..., 3] > 0.0
        image2[face, 3] = 0.0

    # 将arr初始化为原图的副本（image1不会被修改）
    arr = image1.copy()

    # 找出水印中不透明的像素（完全透明的像素将不参与计算）
    opaque_pixels = image2[..., 3] > 0.0

    # 计算叠加模式的结果
    mask = image1[opaque_pixels, :3] < 0.5
    arr[opaque_pixels, :3] = np.where(mask, 2 * image1[opaque_pixels, :3] * image2[opaque_pixels, :3],
                                      1 - 2 * (1 - image1[opaque_pixels, :3]) * (1 - image2[opaque_pixels, :3]))

    # 考虑水印的透明度
    arr[opaque_pixels, :3] = (arr[opaque_pixels, :3] * image2[opaque_pixels, 3, np.newaxis]
                              + image1[opaque_pixels, :3] * (1 - image2[opaque_pixels, 3, np.newaxis]))

    # 将原图的alpha通道复制到结果中
    arr[..., 3] = image1[..., 3]

    return arr

def walk_assistant(walk_path, walk_root, file_name):
    if os.path.samefile(walk_root, walk_path):
        return file_name.replace(' ', '_')
    else:
        return (os.path.relpath(walk_root, walk_path).replace(os.sep, '_') + '_' + file_name).replace(' ', '_')

def sy_info_assistant(sy_info_str, default_sy_config_dict):
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
        default_mode = str(default_sy_config_dict.get('默认图层混合模式', ''))
        if default_mode in modes_normal + modes_overlay:
            mode = default_mode
        else:
            mode = ''

    # 检查默认不透明度
    if mode:
        default_opacity = default_sy_config_dict.get('“正常”混合模式默认水印不透明度', '') \
            if mode in modes_normal else default_sy_config_dict.get('“叠加”混合模式默认水印不透明度', '')
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

def process_image(key, value, sy_info, sy_image, snuggle_path, save_format):
    sy_mode = sy_info[1]

    # 导入原图、擦脸图
    src_image = np.array(Image.open(value[0]).convert('RGBA'), dtype=float) / 255.0
    if len(value) > 1:
        src_image_face = np.array(Image.open(value[1]).convert('RGBA'), dtype=float) / 255.0
    else:
        src_image_face = None

    # 调整水印尺寸（sy_image不会被修改）
    height, width, channels = src_image.shape
    sy_image_resize = cv.resize(sy_image, (width, height))

    # 计算图层混合
    if sy_mode in ['0', 'normal', '正常', '普白']:
        image_snuggle = normal(src_image, sy_image_resize, src_image_face)
    elif sy_mode in ['1', 'overlay', '叠加', '浮雕']:
        image_snuggle = overlay(src_image, sy_image_resize, src_image_face)
    else:
        return

    # 保存贴膜图
    if save_format == 'png':
        image_snuggle = Image.fromarray((image_snuggle * 255.0).astype('uint8'), 'RGBA')
        image_snuggle.save(os.path.join(snuggle_path, '贴膜', sy_info[0], key + '.png'), 'PNG')
    elif save_format == 'jpg':
        image_snuggle = Image.fromarray((image_snuggle[..., :3] * 255.0).astype('uint8'), 'RGB')
        image_snuggle.save(os.path.join(snuggle_path, '贴膜', sy_info[0], key + '.jpg'), 'JPEG', quality=100, optimize=True)
    elif save_format == 'webp':
        image_snuggle = Image.fromarray((image_snuggle * 255.0).astype('uint8'), 'RGBA')
        image_snuggle.save(os.path.join(snuggle_path, '贴膜', sy_info[0], key + '.webp'), 'WEBP', lossless=True)
        # image_snuggle.save(os.path.join(snuggle_path, '贴膜', sy_info[0], key + '.webp'), 'WEBP', quality=90)

def snuggle(snuggle_path, save_format, default_sy_config_dict):
    start_time = time.time()

    src_dict = dict()
    for root, dirs, files in os.walk(os.path.join(snuggle_path, '原图')):
        for file in files:
            if '擦脸' not in file and file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file0 = walk_assistant(os.path.join(snuggle_path, '原图'), root, os.path.splitext(file)[0])
                src_dict[file0] = [os.path.join(root, file)]
        for file in files:
            if '擦脸' in file and file.lower().endswith('.png'):
                file0 = walk_assistant(os.path.join(snuggle_path, '原图'), root, os.path.splitext(file)[0].replace('擦脸', ''))
                if src_dict[file0] is not None:
                    src_dict[file0].append(os.path.join(root, file))

    with ThreadPoolExecutor() as executor:
    # with ThreadPoolExecutor(max_workers=int(os.cpu_count()*0.9)) as executor:
        futures = []

        for root, dirs, files in os.walk(os.path.join(snuggle_path, '水印')):
            for file in files:
                if file.lower().endswith('.png'):
                    sy_info = sy_info_assistant(os.path.splitext(file)[0], default_sy_config_dict)
                    if len(sy_info) != 3:
                        continue

                    if not os.path.exists(os.path.join(snuggle_path, '贴膜', sy_info[0])):
                        os.makedirs(os.path.join(snuggle_path, '贴膜', sy_info[0]))

                    # 导入水印
                    sy_image = Image.open(os.path.join(root, file)).convert('RGBA')
                    sy_image = np.array(sy_image, dtype=float) / 255.0

                    # 调整水印不透明度
                    if 0.0 <= float(sy_info[2]) <= 100.0:
                        sy_opacity = float(sy_info[2]) / 100.0
                    else:
                        continue
                    sy_image[..., 3] *= sy_opacity

                    for key, value in src_dict.items():
                        futures.append(executor.submit(process_image, key, value, sy_info, sy_image, snuggle_path, save_format))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="贴膜进度"):
            pass

    end_time = time.time()
    elapsed_time = int(end_time - start_time)
    print(f"贴膜完成，用时{elapsed_time}秒。")

def main():
    config = dict()
    if os.path.exists(os.path.join(os.getcwd(), '贴膜.yaml')):
        with open(os.path.join(os.getcwd(), '贴膜.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    else:
        with open(os.path.join(os.getcwd(), '贴膜.yaml'), 'w', encoding='utf-8') as file:
            content = textwrap.dedent("""
                            贴膜路径: 
                            出图格式: 

                            # 示例：
                            # 贴膜路径: D:\\定制贴膜
                            # 出图格式: png

                            # 说明：
                            # 可以在这里填写默认的贴膜路径和出图格式。（冒号后的空格不要删）
                            # 出图格式选填 png/jpg/webp 。
                            # 这里填了之后就不用每次贴膜前填了（填错了就还是会每次都问）。
                            
                            默认图层混合模式: 
                            “正常”混合模式默认水印不透明度: 
                            “叠加”混合模式默认水印不透明度: 

                            # 示例：
                            # 默认图层混合模式: 1
                            # “正常”混合模式默认水印不透明度: 10
                            # “叠加”混合模式默认水印不透明度: 20

                            # 说明：
                            # 这里还可以设置默认的图层混合模式和水印不透明度。
                            # 图层混合模式选填0或1，0为“正常”，1为“叠加”。直接写中文字也可以。
                            # 水印不透明度选填0-100之间的数，0为完全透明，100为完全不透明。
                            # 不是每一项都必须填的，按需要填。
                            
                            # 脚本使用说明：
                            # 贴膜脚本使用python语言的pillow图像处理库和numpy数学库制作。只做了“正常”和“叠加”两种图层混合模式。
                            # 首次运行会自动在同文件夹下创建本yaml文件，并在贴膜路径创建“原图”、“水印”、“贴膜”三个文件夹。
                            # 将要贴的原图放在“原图”文件夹，将要贴的水印图片放在“水印”文件夹。文件夹中可以有多层文件夹。水印应按规则命名。
                            # 如果需要擦脸，需要提前在ps等软件中将擦脸部位另存一张png图片，命名为原图文件名加“擦脸”两个字，放在原图同文件夹中。
                            # 贴膜时，会一次性贴“水印”文件夹中的所有水印和“原图”文件夹中的所有原图。不需要贴的图或水印请放在别的地方。
                            # 可以按水印命名规则为每个水印指定贴膜时用的图层混合模式和不透明度。如果有错误，这张水印就不贴。
                            # 贴膜完成后，可以在“贴膜”文件夹中找到每个水印对应的贴膜图。
                            # 仅支持8位图（有的地方会显示为24位或32位）。

                            # 水印命名规则：
                            # 水印命名格式为：标签-模式-不透明度.png
                            # 标签可以是贴膜QQ号或者其他一些用于区分的内容。
                            # 模式选填0或1，0为“正常”，1为“叠加”。直接写中文字也可以。
                            # 不透明度为2-100之间的数。（这是为了和图层混合模式占用的0和1两个数字区分）
                            # 如果已经在本yaml文件中填写了默认信息的话，可以不用在每个水印文件名中都写模式和不透明度。
                            # 例如，水印文件名可以为：标签.png、标签-不透明度.png、标签-模式.png。
                            # 但是如果本yaml文件中的默认参数没填或填错，水印也没按规则命名的话，这张水印就整个不贴。

                            # 水印命名示例：
                            # 10100-0-10.png  （“正常”模式，10%不透明度）
                            # 10200-1-20.png  （“叠加”模式，20%不透明度）
                            # 20100-叠加.png  （“叠加”模式，不透明度按本文件中填写的默认值，没填或填错这个水印就不贴）
                            # 20200-10.png  （10%不透明度，图层混合模式按本文件中填写的默认值，没填或填错这个水印就不贴）
                            # 300200100.png  （图层混合模式和不透明度都按本文件中填写的默认值，没填或填错这个水印就不贴）
                        """).strip()
            file.write(content)

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

        sy_config = dict()
        for item in ['默认图层混合模式', '“正常”混合模式默认水印不透明度', '“叠加”混合模式默认水印不透明度']:
            if config.get(item, "") is not None:
                sy_config[item] = config.get(item, "")
            else:
                sy_config[item] = ""
    except:
        return

    for dir in ['原图', '水印', '贴膜']:
        if not os.path.exists(os.path.join(PATH, dir)):
            os.makedirs(os.path.join(PATH, dir))

    while True:
        try:
            input("按Enter键开始贴膜~")
            snuggle(PATH, SAVEFORMAT, sy_config)
            print('')
        except:
            break

if __name__ == "__main__":
    main()





