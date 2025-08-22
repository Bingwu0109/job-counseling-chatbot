# TYPE_CHECKING是一个布尔值，当运行类型检查工具（如mypy）时，它被设置为True，但在正常Python运行时是False。
# 这用于类型注解和条件导入，以避免在运行时产生不必要的依赖。
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # 导入不同的OCR库
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    ''' 根据环境条件动态地选择和初始化OCR识别引擎的实现
    use_cuda : 是否使用CUDA加速（对于支持CUDA的OCR引擎）
    '''
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        # 返回创建的RapidOCR实例
        ocr = RapidOCR()
    return ocr
