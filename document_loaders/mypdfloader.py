from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import cv2
# Image类是Pillow库中用于图像处理的核心类，支持打开、操作和保存许多不同格式的图像文件。
from PIL import Image
import numpy as np
from configs import PDF_OCR_THRESHOLD
from document_loaders.ocr import get_ocr
import tqdm


class RapidOCRPDFLoader(UnstructuredFileLoader):
    '''一个专门用于处理PDF文件的加载器类，通过OCR技术提取PDF中的文本内容。'''
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            '''
            将给定的图像按照指定的角度进行旋转

            img   --image 待旋转的图像
            angle --rotation angle 旋转角度
            return--rotated img
            '''
            # 获取图像的高度和宽度
            h, w = img.shape[:2]
            # 计算图像旋转的中心点，这是通过将宽度和高度各自除以2来得到的。
            rotate_center = (w/2, h/2)
            # 获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # 计算旋转后图像的新宽度和新高度
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 对旋转矩阵M进行调整，以考虑图像旋转后的平移。
            # 目的是确保旋转后的图像能够完整显示，而不是部分被裁剪掉。
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            # 使用cv2.warpAffine函数和旋转矩阵M对原图像进行变换，得到旋转后的图像rotated_img。
            # 这里指定了变换后的新尺寸(new_w, new_h)
            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img

        def pdf2text(filepath):
            '''将PDF文件转换为文本内容，同时处理PDF中的图像，用OCR（光学字符识别）处理。'''
            # fitz模块，这实际上是PyMuPDF库的一部分，一个功能强大的库，用于处理PDF文件和其他文档。
            import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            # 初始化一个OCR引擎或服务的接口，用于后续的图像文字识别任务。
            ocr = get_ocr()
            # 打开指定路径的PDF文件
            doc = fitz.open(filepath)
            # 存储从PDF文档中提取的全部文本内容
            resp = ""
            # 使用tqdm库创建一个进度条对象b_unit
            # 这个进度条用于可视化地展示PDF文档处理的进度，其中total=doc.page_count设置进度条的总步数为PDF文档的页数，
            # desc参数用于设置进度条的描述信息。
            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            # 遍历PDF文档的每一页
            for i, page in enumerate(doc):
                # 更新进度条的描述信息，显示当前正在处理的页面索引。
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                # 刷新进度条的显示
                b_unit.refresh()
                # 提取当前页的文本内容
                text = page.get_text("")
                # 将提取的文本内容追加到resp字符串中，并在每页的内容后添加一个换行符，以便区分各页的内容。
                resp += text + "\n"
                # 获取当前页面中的图像信息，其中xrefs=True参数指示方法返回图像的交叉引用信息。
                # 这个步骤是准备将页面上的图像进行OCR处理的前置操作
                img_list = page.get_image_info(xrefs=True)
                # 历之前获取的当前PDF页面的所有图像信息列表
                for img in img_list:
                    # 从图像信息中获取xref（交叉引用表的引用号）
                    if xref := img.get("xref"):
                        # 获取图像的边界框，它是一个四元组，包含图像在页面上的位置和尺寸信息。
                        bbox = img["bbox"]
                        # 检查图像的尺寸是否超过设定的阈值PDF_OCR_THRESHOLD
                        # 如果图像太小，则跳过OCR处理，以节省资源。
                        if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                            continue
                        # 使用fitz.Pixmap创建图像的像素映射，它接收文档对象和图像的xref作为参数。
                        pix = fitz.Pixmap(doc, xref)
                        # 获取像素映射中的样本数据
                        samples = pix.samples
                        # 如果页面有旋转（page.rotation不为0），则将图像旋转回原始方向，以确保OCR处理的准确性。
                        if int(page.rotation)!=0:
                            # 首先，将fitz.Pixmap对象pix的像素数据samples转换成一个NumPy数组img_array。
                            # np.frombuffer直接将字节数据转换为NumPy数组，不进行复制。
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                            # 将NumPy数组转换成一个PIL.Image对象tmp_img
                            tmp_img = Image.fromarray(img_array);
                            # PIL.Image对象转换回NumPy数组，并使用cv2.cvtColor将图像从RGB颜色空间转换到BGR颜色空间，
                            # 因为OpenCV主要使用BGR格式。
                            ori_img = cv2.cvtColor(np.array(tmp_img),cv2.COLOR_RGB2BGR)
                            # 调用函数对图像进行旋转，旋转角度是360-page.rotation，这样可以将旋转过的页面图像旋转回原始方向。
                            rot_img = rotate_img(img=ori_img, angle=360-page.rotation)
                            # 将旋转后的图像rot_img从RGB转换回BGR颜色空间，准备用于OCR识别。
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            # 如果页面没有旋转，则直接从pix.samples创建图像数组，过程与旋转情况下的第一步相同。
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        # 调用OCR函数，传入处理后的图像数组img_array，尝试提取图像中的文本。
                        result, _ = ocr(img_array)
                        # 如果OCR处理有结果
                        if result:
                            # 从OCR结果中提取文本行，假设每个元素的第二个位置（索引为1）是文本内容。
                            ocr_result = [line[1] for line in result]
                            # 将提取的文本行以换行符连接
                            resp += "\n".join(ocr_result)

                # 在处理完当前页面的所有图像后，更新进度条，进度增加1。
                b_unit.update(1)
            # 在处理完所有页面后，返回累积的响应字符串，包含了从PDF文档中提取的所有文本内容。
            return resp

        # 调用定义的函数
        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        # 文本分块
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRPDFLoader(file_path="/Users/tonysong/Desktop/test.pdf")
    docs = loader.load()
    print(docs)
