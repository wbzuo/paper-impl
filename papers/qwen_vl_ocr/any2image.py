import io
import os
import time
from pdf2image import convert_from_path
from PIL import Image
import fitz
 
 
def pdf_to_long_png(pdf_path, output_path="output.png", dpi=100, spacing=0):
    """
    将PDF转换为长图PNG（使用PyMuPDF）
    
    参数:
        pdf_path: PDF文件路径
        output_path: 输出PNG文件路径
        dpi: 图像分辨率，默认200
        spacing: 页面间距，默认20像素，设为0则无间距
    """
    try:
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            print(f"错误：PDF文件不存在 - {pdf_path}")
            return False
        
        
        # 打开PDF文档
        doc = fitz.open(pdf_path)
        
        if len(doc) == 0:
            print("错误：PDF文档为空")
            doc.close()
            return False
        
        images = []
        
        # 逐页转换为图像
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 设置转换矩阵（控制DPI）
            mat = fitz.Matrix(dpi/72, dpi/72)  # 72是默认DPI
            
            # 将页面转换为像素图
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        doc.close()
        
        # 计算总尺寸
        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights) + spacing * (len(images) - 1)
        
        # 创建新图像（白色背景）
        new_im = Image.new('RGB', (max_width, total_height), color='white')
        
        # 拼接图像
        y_offset = 0
        for i, image in enumerate(images):
            # 居中对齐
            x_offset = (max_width - image.size[0]) // 2
            new_im.paste(image, (x_offset, y_offset))
            y_offset += image.size[1] + spacing
        
        # 保存最终图像
        new_im.save(output_path, 'PNG', optimize=True, quality=95)
        print(f"成功生成长图: {output_path}")
        print(f"尺寸: {max_width} x {total_height} 像素")
        print(f"包含 {len(images)} 页")
        
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False
 


if __name__ == "__main__":
    # 使用方法
    source_path = r'C:\Users\Administrator\Desktop\251\项目\train_20200121\resume_train_20200121\pdf'
    target_path = r'C:\Users\Administrator\Desktop\251\项目\train_20200121\resume_train_20200121\images'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    
    # 遍历文件
    st = time.time()
    processed_count = 0
    error_count = 0
    for file_name in os.listdir(source_path):
        pdf_path = os.path.join(source_path, file_name)
        image_path = os.path.join(target_path, file_name.replace('pdf', 'png'))
        print(f"正在处理: {file_name}")
        if pdf_to_long_png(pdf_path, image_path):
            processed_count += 1
        else:
            error_count += 1
    ed = time.time()
    print(f"\n处理完成!")
    print(f"总耗时: {ed - st:.2f} 秒")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    # pdf_path = r'C:\Users\Administrator\Desktop\251\项目\train_20200121\resume_train_20200121\pdf\0a2df74bbc31.pdf'  # 你的PDF文件路径
    # output_path = r'C:\Users\Administrator\Desktop\251\项目\train_20200121\resume_train_20200121\0a2df74bbc31.png'  # 输出图片路径
    
    # # 方法1：使用简化版本（推荐）
    # success = pdf_to_long_png(pdf_path, output_path)
    
    # if success:
    #     print("PDF合并为图片成功！")
    # else:
    #     print("转换失败！")