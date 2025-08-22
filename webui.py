import streamlit as st
from webui_pages.utils import *
# 这个库提供了额外的菜单选项功能，使得在Streamlit应用中创建导航菜单变得更简单。
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from configs import VERSION
from server.utils import api_address

# 创建一个ApiRequest对象，用于发起网络请求
api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv
    # 配置Streamlit页面的标题、图标、侧边栏状态和菜单项
    st.set_page_config(
        "Langchain-Chatchat WebUI",
        os.path.join("img", "chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About': f"""Welcome Career Counseling Chatbot {VERSION}！"""
        }
    )

    pages = {
        "dialogue": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "Knowledge Base Management": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    # 创建一个侧边栏区域
    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "Chatbot.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">Current version：{VERSION}</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0

        # 使用option_menu创建一个选项菜单，用户可以通过这个菜单选择不同的页面。
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
