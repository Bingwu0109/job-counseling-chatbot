from langchain.docstore.document import Document
import re


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """用于检查一段文本中非字母字符的比例是否超过了给定的阈值。
    这个功能主要用于识别那些不适合被标记为标题或叙述文本的字符串，例如连续的破折号之类的文本。

    Parameters
    ----------
    text
        输入的文本
    threshold
        表示非字母字符比例的阈值
    """
    # 检查输入文本的长度是否为0，即文本是否为空。
    if len(text) == 0:
        return False
    # 计算文本中字母字符的数量，
    # 通过列表推导式选出既不是空格（char.strip()为True）又是字母（char.isalpha()为True）的字符，
    # 然后计算这样的字符的总数。
    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    # 计算文本中非空格字符的总数
    total_count = len([char for char in text if char.strip()])
    try:
        # 计算字母字符占非空格字符的比例
        ratio = alpha_count / total_count
        # 如果计算出的比例小于给定的阈值，则返回True；否则返回False。
        return ratio < threshold
    except:
        return False


def is_possible_title(
        text: str, # 要检查的文本字符串
        title_max_word_length: int = 20, # 标题可以包含的最大单词数量
        non_alpha_threshold: float = 0.5, # 文本被认为是标题所需的最小字母字符比例，默认值为0.5。
) -> bool:
    """用来检查一段文本是否可能是一个有效的标题
    """

    # 文本长度为0的话，肯定不是title
    if len(text) == 0:
        print("Not a title. Text is empty.")
        return False

    # 文本中有标点符号，就不是title
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    # 如果文本以标点符号结束，则返回False，因为标题通常不以标点符号结束。
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    # 检查文本拆分成单词后的数量是否超过了允许的最大值。如果是，返回False。
    if len(text) > title_max_word_length:
        return False

    # 检查非字母字符的比例是否过高。如果是，返回False。
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # 检查文本是否以任何标点符号结束。这是为了防止像"亲爱的朋友们,"这样的称呼被标记为标题。
    if text.endswith((",", ".", "，", "。")):
        return False
    # 检查文本是否全部由数字组成
    if text.isnumeric():
        print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # 如果文本长度小于5，就使用整个文本进行检查。
    if len(text) < 5:
        text_5 = text
    else:
        # 否则，只检查前5个字符。
        text_5 = text[:5]
    # 计算前5个字符中数字字符的数量
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    # 如果前5个字符中没有数字，返回False。
    if not alpha_in_text_5:
        return False

    return True


def zh_title_enhance(docs: Document) -> Document:
    '''处理一个Document对象的列表（docs），通过判断其中的文档内容是否可能是一个标题，来增强处理中文标题的能力。'''
    # 临时存储被识别为标题的文本内容
    title = None
    # 检查传入的docs列表是否非空
    if len(docs) > 0:
        # 遍历docs列表中的每一个Document对象doc
        for doc in docs:
            # 检查doc的page_content（即页面内容）是否可能是一个标题
            if is_possible_title(doc.page_content):
                # 将doc的metadata['category']设置为'cn_Title'，表明这个Document对象被分类为含有中文标题。
                doc.metadata['category'] = 'cn_Title'
                title = doc.page_content
            elif title:
                # 修改当前文档的page_content，在其内容前添加一段说明文字，说明这部分内容与之前识别的标题title有关。
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")
