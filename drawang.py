from PIL import Image, ImageDraw, ImageFont
import os

def add_text_bottom_left(
    img_path,
    out_path,
    text,
    font_path,
    font_size_ratio=0.038,   # 字体大小占图像高度比例，可调
    margin_ratio=0.03,       # 边距占图像尺寸比例，可调
    text_color=(255, 0, 0)   # 红色
):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    font_size = int(h * font_size_ratio)
    margin = int(min(w, h) * margin_ratio)

    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img)

    # 计算多行文字尺寸
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 左下角位置
    x = margin
    y = h - text_h - margin

    # 直接绘制红色文字
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        spacing=4
    )

    img.save(out_path)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    # 改成你自己的字体路径
    # Windows 示例:
    # font_path = r"C:\Windows\Fonts\simhei.ttf"
    # Ubuntu 示例:
    # font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    font_path = r"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

    # 四张图路径
    image_paths = [
        "experient_fig/track/1.png",
        "experient_fig/track/1.png",
        "experient_fig/track/1.png",
        "experient_fig/track/1.png"
    ]

    # 每张图对应要打印的文字
    texts = [
        "前端1\n方位 268.33   俯仰 -18.53",
        "前端2\n方位 272.12   俯仰 -17.48",
        "前端3\n方位 277.91   俯仰 -15.60",
        "前端4\n方位 269.45   俯仰 -18.51"
    ]

    # 输出文件夹
    os.makedirs("output", exist_ok=True)

    for i, (img_path, text) in enumerate(zip(image_paths, texts), start=1):
        out_path = os.path.join("output", f"img{i}_text.png")
        add_text_bottom_left(
            img_path=img_path,
            out_path=out_path,
            text=text,
            font_path=font_path
        )