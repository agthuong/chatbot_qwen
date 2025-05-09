import streamlit as st
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import json5
import os
import datetime
import json
import re
# import time # Streamlit xử lý timing khác
# import unidecode # Không cần cho RAG siêu đơn giản
# from sklearn.feature_extraction.text import TfidfVectorizer # Không embedding
# from sklearn.metrics.pairwise import cosine_similarity # Không search
# import numpy as np # Không cần numpy
# from fuzzywuzzy import fuzz # Có thể không cần thiết nữa

# --- Các biến toàn cục cho cấu hình ---
LLM_CFG = {
    'model': 'qwen1.5-32b-chat-q4_0',
    'model_server': 'http://localhost:1234/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {'temperature': 0.1}
}

RAW_RAG_SYSTEM_PROMPT = """Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TRÊN NỘI DUNG ĐƯỢC CUNG CẤP. 
Toàn bộ nội dung tài liệu liên quan sẽ được cung cấp cho bạn ngay sau câu "Dưới đây là nội dung tài liệu:". 
Hãy đọc kỹ tài liệu này và câu hỏi của người dùng để đưa ra câu trả lời chính xác nhất. 
**Quan trọng: Cố gắng hiểu ý định thực sự của người dùng, ngay cả khi câu hỏi có vẻ chứa lỗi chính tả hoặc từ viết tắt không rõ ràng. Ví dụ, nếu người dùng hỏi 'giai doan cuoi amf gi', hãy xem xét khả năng họ muốn hỏi 'giai đoạn cuối làm gì' hoặc 'giai đoạn cuối là gì' và tìm thông tin tương ứng trong tài liệu.**
Chỉ trả lời dựa trên thông tin có trong tài liệu. Nếu thông tin không có trong tài liệu, hãy nói rõ "Thông tin này không có trong tài liệu được cung cấp.".
Trình bày câu trả lời một cách rõ ràng và súc tích.
"""

# Biến toàn cục để theo dõi trạng thái đăng ký tools
TOOLS_REGISTERED = False

# --- Định nghĩa các class tool (không đăng ký ở đây) ---
class TraCuuQuyTrinhTextRawTool(BaseTool):
    description = 'Cung cấp toàn bộ nội dung từ file văn bản để LLM tự tìm kiếm thông tin trả lời câu hỏi.'
    parameters = [
        {'name': 'trigger', 'type': 'string', 'description': 'Một chuỗi bất kỳ để kích hoạt tool lấy nội dung file.', 'required': False }
    ]
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data_file_path = './test_data.txt'
        self.full_text_content = self._load_full_text_content(self.data_file_path)
        if self.full_text_content is None:
            st.warning(f"Tool: LỖI - Không thể tải nội dung từ {self.data_file_path}.")

    def _load_full_text_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            return content
        except FileNotFoundError: st.error(f"Tool: File not found {file_path}"); return None
        except Exception as e: st.error(f"Tool: Error reading file {file_path}: {e}"); return None

    def call(self, params: str, **kwargs):
        if self.full_text_content is not None:
            return {'success': True, 'full_document_content': self.full_text_content}
        else:
            return {'success': False, 'error': f'Tool: Không thể tải nội dung từ file {self.data_file_path}.'}

class TraCuuQuyTrinhCSVTool(BaseTool):
    description = 'Tra cứu thông tin quy trình công việc từ CSDL CSV. Cung cấp thông tin chính xác về các quy trình của công ty.'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': 'Câu hỏi về quy trình cần tìm. Có thể bao gồm từ khóa về giai đoạn, phòng ban, công việc, người thực hiện.',
        'required': True
    }]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data_file_path = './data.csv'
        try:
            # Chỉ kiểm tra xem file có tồn tại không, không đọc hết nội dung
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                self.file_exists = True
        except FileNotFoundError:
            st.error(f"Tool: File CSV not found {self.data_file_path}")
            self.file_exists = False
        except Exception as e:
            st.error(f"Tool: Error checking CSV file {self.data_file_path}: {e}")
            self.file_exists = False

    def call(self, params: str, **kwargs):
        if not self.file_exists:
            return {'success': False, 'error': f'Tool: Không thể tải nội dung từ file {self.data_file_path}.'}
        
        # Phân tích tham số đầu vào
        try:
            if isinstance(params, str) and not params.startswith('{'):
                query = params
            else:
                query_data = json5.loads(params)
                query = query_data['query']
        except Exception as e:
            query = str(params)
        
        try:
            # Đọc một phần nhỏ của file để kiểm tra
            import pandas as pd
            df_preview = pd.read_csv(self.data_file_path, nrows=5)
            
            # Trả về thông tin giản lược
            return {
                'success': True,
                'message': f'Đã tìm thấy thông tin cho "{query}" trong file CSV. Đây là dữ liệu cấu trúc với {len(df_preview.columns)} cột.',
                'preview': df_preview.to_dict(orient='records')[:2]  # Chỉ trả về 2 hàng đầu để minh họa
            }
        except Exception as e:
            return {'success': False, 'error': f'Lỗi khi đọc file CSV: {str(e)}'}

# --- Hàm để lấy hoặc tạo tool và assistant dựa trên dataset được chọn ---
@st.cache_resource
def get_chatbot_essentials(dataset_choice):
    """
    Khởi tạo và trả về assistant phù hợp với dataset được chọn
    
    Args:
        dataset_choice: Chuỗi chỉ định dataset được chọn ("text" hoặc "csv")
    
    Returns:
        Tuple (assistant, document_content)
    """
    global TOOLS_REGISTERED
    
    # Đăng ký tools nếu chưa được đăng ký
    if not TOOLS_REGISTERED:
        # Đăng ký tools với allow_overwrite=True để tránh lỗi nếu đã tồn tại
        text_tool_cls = register_tool('tra_cuu_quy_trinh_text_raw', allow_overwrite=True)(TraCuuQuyTrinhTextRawTool)
        csv_tool_cls = register_tool('tra_cuu_quy_trinh_csv', allow_overwrite=True)(TraCuuQuyTrinhCSVTool)
        TOOLS_REGISTERED = True
        st.success("Tools đã được đăng ký thành công!")
    
    # Lựa chọn function tương ứng với dataset
    if dataset_choice == "text":
        function_list = ["tra_cuu_quy_trinh_text_raw"]
        # Lấy nội dung của file text
        tool_instance = TraCuuQuyTrinhTextRawTool()
        document_content = tool_instance.full_text_content
    else:  # "csv"
        function_list = ["tra_cuu_quy_trinh_csv"]
        # Với file CSV, chúng ta chỉ kiểm tra xem file có tồn tại không
        tool_instance = TraCuuQuyTrinhCSVTool()
        if tool_instance.file_exists:
            document_content = "File CSV đã được tải. Dữ liệu sẽ được truy xuất khi cần."
        else:
            document_content = None
            st.error("Không thể tìm thấy file data.csv. Vui lòng kiểm tra đường dẫn.")
    
    # Khởi tạo assistant với function list tương ứng
    assistant = Assistant(
        llm=LLM_CFG,
        function_list=function_list,
        system_message=RAW_RAG_SYSTEM_PROMPT
    )
    
    if document_content is None:
        st.error(f"LỖI NGHIÊM TRỌNG: Không thể tải nội dung tài liệu cho dataset {dataset_choice}.")
    
    return assistant, document_content

# Hàm xử lý <think> tag để hiển thị trong collapse box
def format_think_tags(text):
    """Chuyển đổi <think>...</think> thành định dạng details/summary của HTML"""
    pattern = r'<think>(.*?)</think>'
    
    def replace_func(match):
        think_content = match.group(1).strip()
        return f"""
<details>
<summary>💭 Suy nghĩ của AI (nhấn để xem/ẩn)</summary>
<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
{think_content}
</div>
</details>
"""
    
    # Thay thế tất cả các thẻ <think> trong văn bản
    formatted_text = re.sub(pattern, replace_func, text, flags=re.DOTALL)
    return formatted_text

# Kiểm tra xem chunk có chứa thẻ <think> hay không
def chunk_contains_think_tag(chunk):
    return '<think>' in chunk or '</think>' in chunk

# --- Hàm chính cho ứng dụng Streamlit ---
def run_chatbot_app():
    st.title("🤖 Chatbot Quy Trình (RAW RAG)")

    # Bật HTML rendering để hiển thị details/summary
    st.markdown("""
    <style>
    details {
        border: 1px solid #aaa;
        border-radius: 4px;
        padding: .5em .5em 0;
        margin-bottom: 1em;
    }
    summary {
        font-weight: bold;
        cursor: pointer;
        padding: .5em;
        color: #4b8bf4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Sidebar để cấu hình ---
    with st.sidebar:
        st.header("Cấu hình Chatbot")
        
        # Radio button để chọn dataset
        if "dataset_choice" not in st.session_state:
            st.session_state.dataset_choice = "text"  # Mặc định là dataset text
            
        dataset_choice = st.radio(
            "Chọn nguồn dữ liệu:",
            options=["text", "csv"],
            format_func=lambda x: "File Text (test_data.txt)" if x == "text" else "File CSV (data.csv)",
            key="dataset_choice"
        )
        
        st.info(f"Đã chọn dataset: {dataset_choice}")
        
        # Nút để xóa lịch sử chat
        if st.button("Xóa lịch sử chat"):
            st.session_state.messages = []
            st.success("Đã xóa lịch sử chat!")

    # Khởi tạo assistant dựa vào lựa chọn dataset
    assistant, main_document_content = get_chatbot_essentials(st.session_state.dataset_choice)

    if main_document_content is None:
        st.error("Không thể khởi động chatbot do lỗi tải tài liệu.")
        return
    
    if assistant is None:
        st.error("Không thể khởi động chatbot do lỗi khởi tạo Assistant.")
        return

    # Khởi tạo lịch sử chat nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Sử dụng unsafe_allow_html để hiển thị HTML
            st.markdown(message["content"], unsafe_allow_html=True)

    # Xử lý input từ người dùng
    if prompt := st.chat_input("Câu hỏi của bạn về tài liệu?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Thêm ngữ cảnh vào prompt dựa trên loại dataset
        if st.session_state.dataset_choice == "text":
            user_message_with_context = f"{prompt}\n\n---\nDưới đây là nội dung tài liệu:\n{main_document_content}"
        else:  # csv
            user_message_with_context = prompt  # Không cần thêm ngữ cảnh vì tool sẽ truy cập file CSV trực tiếp
        
        prepared_messages_for_llm = []
        if len(st.session_state.messages) > 1:
            prepared_messages_for_llm.extend(st.session_state.messages[:-1])
        prepared_messages_for_llm.append({'role': 'user', 'content': user_message_with_context})

        with st.chat_message("assistant"):
            raw_full_response = ""  # Lưu trữ phản hồi nguyên bản trước khi xử lý
            buffer = ""  # Buffer để giữ nội dung chưa hoàn chỉnh
            think_blocks = []  # Lưu các khối <think> đã xử lý để tránh lặp lại
            visible_content_placeholder = st.empty()  # Placeholder cho nội dung hiển thị
            
            # Sử dụng một spinner đơn giản nếu muốn có chỉ báo loading
            with st.spinner("🤖 LLM đang suy nghĩ..."):
                try:
                    # Generator này sẽ xử lý từng chunk và trích xuất phần nội dung mới
                    def generate_responses_for_streaming_and_history():
                        nonlocal raw_full_response, buffer, think_blocks
                        assistant_responded_flag = False
                        previous_text = ""  # Lưu toàn bộ văn bản trước đó
                        
                        for r_chunk_list in assistant.run(messages=prepared_messages_for_llm):
                            if r_chunk_list:
                                last_message_in_chunk = r_chunk_list[-1]
                                if last_message_in_chunk.get('role') == 'assistant':
                                    current_chunk = last_message_in_chunk.get('content', '')
                                    if current_chunk:  # Chỉ xử lý nếu có nội dung
                                        # Tìm phần nội dung mới (delta) giữa chunk hiện tại và văn bản trước đó
                                        if current_chunk.startswith(previous_text) and current_chunk != previous_text:
                                            # Chỉ lấy phần mới được thêm vào
                                            new_content = current_chunk[len(previous_text):]
                                            if new_content:  # Chỉ yield nếu có nội dung mới
                                                raw_full_response += new_content
                                                buffer += new_content
                                                previous_text = current_chunk
                                                yield (buffer, False)  # False = không phải nội dung cuối cùng
                                        else:
                                            # Nếu không phải là tiếp nối của văn bản trước
                                            # (hiếm khi xảy ra, nhưng để phòng ngừa)
                                            raw_full_response = current_chunk
                                            buffer = current_chunk
                                            previous_text = current_chunk
                                            yield (buffer, False)
                                    assistant_responded_flag = True
                        
                        if not assistant_responded_flag and not raw_full_response:
                            fallback_msg = "Xin lỗi, tôi không thể tạo phản hồi dựa trên thông tin được cung cấp."
                            raw_full_response = fallback_msg
                            buffer = fallback_msg
                            yield (buffer, True)  # True = nội dung cuối cùng
                        else:
                            yield (buffer, True)  # Phát ra nội dung cuối cùng
                    
                    # Xử lý từng chunk được trả về
                    for content, is_final in generate_responses_for_streaming_and_history():
                        # Xử lý và hiển thị nội dung tạm thời
                        formatted_content = format_think_tags(content)
                        visible_content_placeholder.markdown(formatted_content, unsafe_allow_html=True)
                    
                    # Sau khi stream kết thúc, xử lý nội dung đầy đủ
                    full_processed_response = format_think_tags(raw_full_response)

                except Exception as e:
                    st.error(f"Lỗi khi chạy assistant: {e}")
                    full_processed_response = "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn."
            
            # Cập nhật nội dung cuối cùng
            visible_content_placeholder.markdown(full_processed_response, unsafe_allow_html=True)
            
            # Lưu vào lịch sử cuộc trò chuyện
            st.session_state.messages.append({"role": "assistant", "content": full_processed_response})

if __name__ == "__main__":
    run_chatbot_app()

# Để chạy ứng dụng này:
# 1. Đảm bảo bạn đã cài đặt streamlit: pip install streamlit (version >= 1.29.0 để có st.write_stream)
# 2. Lưu file này (ví dụ: chatbot_qwen/run_local_from_text.py)
# 3. Chạy từ terminal: streamlit run chatbot_qwen/run_local_from_text.py