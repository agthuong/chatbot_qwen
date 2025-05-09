from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import pandas as pd
import json5
import os
import datetime
import json
import time
from fuzzywuzzy import fuzz  # Thêm thư viện so sánh mờ

# 🔧 Import tool định nghĩa ở file khác nếu có
# Ví dụ: from tools import csv_tool

@register_tool('tra_cuu_quy_trinh')
class TraCuuQuyTrinhTool(BaseTool):
    description = 'Tra cứu thông tin quy trình công việc từ CSDL. Cung cấp thông tin chính xác về các quy trình của công ty.'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': 'Câu hỏi về quy trình cần tìm. Có thể bao gồm từ khóa về giai đoạn, phòng ban, công việc, người thực hiện.',
        'required': True
    }]

    def __init__(self, *args, **kwargs):
        super().__init__()
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re
        import unidecode  # Thêm thư viện xử lý dấu tiếng Việt
        
        # Tải dữ liệu
        self.df = pd.read_csv('./data.csv')
        
        # In thông tin dạng dữ liệu để debug
        print("Tên các cột trong file CSV:", self.df.columns.tolist())
        
        # Kiểm tra xem có cột 'Unnamed' nào không
        has_unnamed_columns = any('Unnamed' in col for col in self.df.columns)
        
        # Phân tích xem file CSV sử dụng tên cột thật hay Unnamed
        if has_unnamed_columns:
            # Phương pháp cũ nếu có cột Unnamed
            self.use_unnamed_columns = True
            try:
                header_row = self.df.iloc[1]
                actual_headers = {}
                
                # Lấy tên cột thực tế từ dòng header
                for i, col_name in enumerate(self.df.columns):
                    if i < len(header_row) and isinstance(header_row[i], str) and header_row[i].strip():
                        actual_headers[header_row[i]] = col_name
                        print(f"Header thực tế: '{header_row[i]}' ở cột '{col_name}'")
                
                # In thông tin về header đã phân tích
                print(f"Đã phân tích được {len(actual_headers)} header thực tế từ CSV")
            except Exception as e:
                print(f"Lỗi khi phân tích header: {e}")
                actual_headers = {}
                
            # Kiểm tra và loại bỏ các hàng tiêu đề/metadata 
            first_rows = self.df.iloc[:5]
            is_header = []
            
            for i, row in first_rows.iterrows():
                header_like_count = sum(1 for x in row if isinstance(x, str) and 
                    (x.istitle() or x.isupper() or x in ['A', 'R'] or 
                     any(term in x for term in ['Giai đoạn', 'Công việc', 'Phòng ban', 'Mục tiêu'])))
                is_header.append(header_like_count > 3)
            
            # Tạo danh sách các chỉ số cần loại bỏ
            drop_indices = [i for i, is_h in enumerate(is_header) if is_h and i < 5]
            
            # Loại bỏ các hàng header bằng index
            if drop_indices:
                self.df = self.df.drop(drop_indices).reset_index(drop=True)
                
            # Cập nhật ánh xạ tên cột dựa trên dữ liệu thực tế trong file CSV
            self.column_names = {
                'Giai đoạn': actual_headers.get('Giai đoạn', 'Unnamed: 1'),
                'Công việc': actual_headers.get('Công việc', 'Unnamed: 2'),
                'Phòng ban': actual_headers.get('Phòng ban', 'Unnamed: 3'),
                'Điều kiện tiên quyết': actual_headers.get('Điều kiện tiên quyết', 'Unnamed: 4'),
                'Người phụ trách': actual_headers.get('A', 'Unnamed: 5'),  # Cột A
                'Người thực hiện': actual_headers.get('R', 'Unnamed: 6'),  # Cột R
                'Làm gì': actual_headers.get('Làm gì', 'Unnamed: 7'),
                'Mục tiêu': actual_headers.get('Mục tiêu', 'Unnamed: 8'),
                'Kết quả trả ra': actual_headers.get('Kết quả trả ra', 'Unnamed: 9'),
                'Duration': actual_headers.get('Duration', 'Unnamed: 10'),
                'Receiver': actual_headers.get('Receiver', 'Unnamed: 11'),
                'Định kỳ thực hiện': actual_headers.get('Định kỳ thực hiện', 'Unnamed: 12'),
                'Gửi kết quả qua': actual_headers.get('Gửi kết quả qua', 'Unnamed: 13'),
                'Form thực hiện': actual_headers.get('Form thực hiện', 'Unnamed: 14'),
                'Đo lường/đánh giá': actual_headers.get('Đo lường/đánh giá', 'Unnamed: 15')
            }
        else:
            # Phương pháp mới nếu có tên cột thật
            self.use_unnamed_columns = False
            # Lọc dữ liệu trống
            # Kiểm tra xem dòng đầu tiên có phải là header không
            if pd.isna(self.df.iloc[0]['Giai đoạn']) or self.df.iloc[0]['Giai đoạn'] == '':
                self.df = self.df.iloc[1:].reset_index(drop=True)
                
            # Ánh xạ trực tiếp tên cột
            self.column_names = {
                'Giai đoạn': 'Giai đoạn',
                'Công việc': 'Công việc',
                'Phòng ban': 'Phòng ban',
                'Điều kiện tiên quyết': 'Điều kiện tiên quyết',
                'Người phụ trách': 'A',  # Cột A
                'Người thực hiện': 'R',  # Cột R
                'Làm gì': 'Làm gì',
                'Mục tiêu': 'Mục tiêu',
                'Kết quả trả ra': 'Kết quả trả ra',
                'Duration': 'Duration',
                'Receiver': 'Receiver',
                'Định kỳ thực hiện': 'Định kỳ thực hiện',
                'Gửi kết quả qua': 'Gửi kết quả qua',
                'Form thực hiện': 'Form thực hiện',
                'Đo lường/đánh giá': 'Đo lường/đánh giá'
            }
            
        # Lọc dữ liệu trống một cách chặt chẽ hơn
        content_mask = np.array([
            sum(1 for x in row if pd.notna(x) and str(x).strip() != '') > 3 
            for _, row in self.df.iterrows()
        ])
        
        self.df = self.df[content_mask].reset_index(drop=True)
        
        # Sửa lỗi fillna
        for col in self.df.columns:
            if self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64':
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna('', inplace=True)
        
        # In ra thống kê để kiểm tra
        print(f"Dữ liệu sau khi lọc: {len(self.df)} hàng, {len(self.df.columns)} cột")
        
        # In ra ánh xạ tên cột để kiểm tra
        print("Ánh xạ tên cột:")
        for friendly_name, col_name in self.column_names.items():
            print(f"  {friendly_name} -> {col_name}")
        
        # Tạo từ điển map ngược lại để tiện sử dụng
        self.inverse_column_names = {v: k for k, v in self.column_names.items()}
        
        # Hàm tiền xử lý văn bản (làm sạch và chuẩn hóa)
        def preprocess_text(text):
            if not isinstance(text, str):
                return ""
            # Chuyển sang chữ thường
            text = text.lower()
            # Loại bỏ dấu câu và ký tự đặc biệt
            text = re.sub(r'[^\w\s]', ' ', text)
            # Loại bỏ khoảng trắng thừa
            text = re.sub(r'\s+', ' ', text).strip()
            # Tạo phiên bản không dấu để hỗ trợ tìm kiếm không phân biệt dấu
            text_no_accent = unidecode.unidecode(text)
            # Kết hợp cả văn bản gốc và không dấu
            return text + " " + text_no_accent
        
        # Lưu lại hàm tiền xử lý văn bản để sử dụng sau này
        self.preprocess_text = preprocess_text
        
        # Danh sách stopwords tiếng Việt mở rộng
        self.vietnamese_stopwords = [
            'và', 'hoặc', 'của', 'là', 'để', 'trong', 'ngoài', 'khi', 'với', 'bởi', 
            'được', 'không', 'có', 'này', 'đó', 'các', 'những', 'mà', 'nên', 'vì',
            'trên', 'dưới', 'tại', 'từ', 'qua', 'một', 'hai', 'ba', 'bốn', 'năm',
            'sáu', 'bảy', 'tám', 'chín', 'mười', 'rồi', 'sẽ', 'đang', 'đã', 'sắp',
            'ai', 'tôi', 'bạn', 'họ', 'chúng', 'chúng ta', 'mình', 'chiếc', 'cái'
        ]
        
        # Xây dựng ánh xạ vai trò - giai đoạn
        self.build_role_stage_mapping()
        
        # Tạo các trường tìm kiếm theo chủ đề
        self.build_search_fields()
        
    def build_role_stage_mapping(self):
        """Xây dựng ánh xạ tự động giữa người thực hiện và giai đoạn"""
        role_stage_map = {}
        
        for idx, row in self.df.iterrows():
            role_col = self.column_names['Người thực hiện']
            stage_col = self.column_names['Giai đoạn']
            
            if (isinstance(row[role_col], str) and row[role_col].strip() and 
                isinstance(row[stage_col], str) and row[stage_col].strip()):
                
                # Tách các vai trò nếu có nhiều (phân tách bởi dấu phẩy)
                roles = [r.strip().lower() for r in row[role_col].split(',')]
                stage = row[stage_col].strip()
                
                for role in roles:
                    if role not in role_stage_map:
                        role_stage_map[role] = []
                    if stage not in role_stage_map[role]:
                        role_stage_map[role].append(stage)
        
        self.role_stage_mapping = role_stage_map
        
        # In ra để kiểm tra
        print(f"Đã xây dựng ánh xạ vai trò - giai đoạn cho {len(role_stage_map)} vai trò.")

    def build_search_fields(self):
        """Tạo các trường tìm kiếm theo ngữ nghĩa và chỉ mục vector tương ứng cho tất cả cột"""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Tạo các trường tìm kiếm chính
        # 1. Trường nhân sự
        self.df['person_fields'] = self.df.apply(
            lambda row: ' '.join([
                str(row[self.column_names['Người thực hiện']]), 
                str(row[self.column_names['Người phụ trách']])
            ]), axis=1).apply(self.preprocess_text)
        
        # 2. Trường quy trình
        self.df['process_fields'] = self.df.apply(
            lambda row: ' '.join([
                str(row[self.column_names['Giai đoạn']]), 
                str(row[self.column_names['Công việc']]), 
                str(row[self.column_names['Phòng ban']])
            ]), axis=1).apply(self.preprocess_text)
        
        # 3. Trường nội dung
        self.df['content_fields'] = self.df.apply(
            lambda row: ' '.join([
                str(row[self.column_names['Mục tiêu']]), 
                str(row[self.column_names['Làm gì']]), 
                str(row[self.column_names['Kết quả trả ra']])
            ]), axis=1).apply(self.preprocess_text)
            
        # 4. Trường thời gian và tần suất
        self.df['time_fields'] = self.df.apply(
            lambda row: ' '.join([
                str(row[self.column_names['Định kỳ thực hiện']]), 
                str(row[self.column_names['Duration']])
            ]), axis=1).apply(self.preprocess_text)
        
        # 5. Trường thông tin bổ sung
        self.df['additional_fields'] = self.df.apply(
            lambda row: ' '.join([
                str(row[self.column_names.get('Điều kiện tiên quyết', '')]),
                str(row[self.column_names.get('Receiver', '')]),
                str(row[self.column_names.get('Gửi kết quả qua', '')]),
                str(row[self.column_names.get('Form thực hiện', '')]),
                str(row[self.column_names.get('Đo lường/đánh giá', '')])
            ]), axis=1).apply(self.preprocess_text)
        
        # Tạo trường tìm kiếm tổng hợp có trọng số
        self.df['search_text'] = ''
        
        # Nhân trọng số cho các trường khác nhau
        field_weights = {
            'process_fields': 3,    # Trọng số cao nhất cho thông tin quy trình
            'content_fields': 2,    # Trọng số cao cho nội dung
            'person_fields': 1,     # Trọng số bình thường cho người thực hiện
            'time_fields': 2,       # Tăng trọng số cho thông tin thời gian
            'additional_fields': 1  # Trọng số thấp cho thông tin bổ sung
        }
        
        for field, weight in field_weights.items():
            weighted_text = self.df[field].apply(lambda x: ' '.join([x] * weight))
            self.df['search_text'] += ' ' + weighted_text
        
        # Vector hóa trường nhân sự
        self.person_vectorizer = TfidfVectorizer(
            lowercase=True,
            max_df=0.9, min_df=1, 
            ngram_range=(1, 2))
        self.person_matrix = self.person_vectorizer.fit_transform(self.df['person_fields'])
        
        # Vector hóa trường quy trình
        self.process_vectorizer = TfidfVectorizer(
            lowercase=True, 
            max_df=0.9, min_df=1, 
            ngram_range=(1, 3))
        self.process_matrix = self.process_vectorizer.fit_transform(self.df['process_fields'])
        
        # Vector hóa trường nội dung
        self.content_vectorizer = TfidfVectorizer(
            lowercase=True, 
            max_df=0.9, min_df=1, 
            ngram_range=(1, 3),
            stop_words=self.vietnamese_stopwords)
        self.content_matrix = self.content_vectorizer.fit_transform(self.df['content_fields'])
        
        # Vector hóa trường thời gian
        self.time_vectorizer = TfidfVectorizer(
            lowercase=True, 
            max_df=0.9, min_df=1, 
            ngram_range=(1, 2))
        self.time_matrix = self.time_vectorizer.fit_transform(self.df['time_fields'])
        
        # Vector hóa trường thông tin bổ sung
        self.additional_vectorizer = TfidfVectorizer(
            lowercase=True, 
            max_df=0.9, min_df=1, 
            ngram_range=(1, 2))
        self.additional_matrix = self.additional_vectorizer.fit_transform(self.df['additional_fields'])
        
        # Vector hóa từng cột riêng lẻ để tìm kiếm chính xác
        self.column_matrices = {}
        self.column_vectorizers = {}
        
        # Tạo vector cho từng cột trong CSV để tìm kiếm chính xác
        for col_name, actual_col in self.column_names.items():
            if actual_col in self.df.columns:
                # Tiền xử lý văn bản trong cột
                processed_col_text = self.df[actual_col].astype(str).apply(self.preprocess_text)
                
                # Tạo vectorizer cho cột
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    max_df=0.95, min_df=1,
                    ngram_range=(1, 2))
                
                try:
                    # Tạo ma trận vector cho cột
                    matrix = vectorizer.fit_transform(processed_col_text)
                    
                    # Lưu trữ cả vectorizer và ma trận
                    self.column_vectorizers[col_name] = vectorizer
                    self.column_matrices[col_name] = matrix
                    
                    print(f"Đã tạo ma trận vector cho cột '{col_name}'")
                except Exception as e:
                    print(f"Lỗi khi vector hóa cột '{col_name}': {e}")
        
        # Vector hóa trường tổng hợp cho tìm kiếm mặc định
        self.main_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            stop_words=self.vietnamese_stopwords,
            max_features=10000,
            min_df=1,
            max_df=0.95,
            token_pattern=r'\b\w+\b'
        )
        self.main_matrix = self.main_vectorizer.fit_transform(self.df['search_text'])

    def call(self, params: str, **kwargs):
        """Tìm kiếm quy trình dựa trên query"""
        import pandas as pd
        import numpy as np
        import json
        import json5
        from fuzzywuzzy import fuzz
        
        # Phân tích tham số đầu vào
        try:
            # Nếu params đã là chuỗi query, dùng trực tiếp
            if isinstance(params, str) and not params.startswith('{'):
                query = params
            # Nếu params là JSON string, parse nó
            else:
                query_data = json5.loads(params)
                query = query_data['query']
        except Exception as e:
            # Fallback: nếu có lỗi khi parse JSON, coi params là query trực tiếp
            print(f"Lỗi khi xử lý tham số: {e}, dùng params làm query.")
            query = str(params)
        
        # Phân tích truy vấn
        query_info = self.analyze_query(query)
        
        # Thiết lập số lượng kết quả trả về
        max_results = 10 if 'time_focus' in query_info['components'] else 3
        
        # Tìm kiếm đa chiều dựa trên loại truy vấn
        search_results = self.multi_vector_search(query_info, top_n=max_results)
        
        # Đặc biệt xử lý truy vấn về thời gian
        if ('time_focus' in query_info['components'] and not search_results) or ('dinh_ky' in query_info['components']):
            # Tìm kiếm trực tiếp theo định kỳ thực hiện
            time_col = self.column_names['Định kỳ thực hiện']
            direct_results = []
            time_unit = query_info['components'].get('time_unit', '')
            
            # Nếu có đơn vị thời gian cụ thể, tìm trực tiếp
            if time_unit:
                print(f"Tìm kiếm trực tiếp với đơn vị thời gian: {time_unit}")
                for idx in range(len(self.df)):
                    if isinstance(self.df.iloc[idx][time_col], str) and time_unit in self.df.iloc[idx][time_col].lower():
                        # Tính điểm số cơ bản cộng thêm điểm cho số từ khóa khớp
                        keyword_matches = sum(1 for kw in query_info['components']['keywords'] 
                                             if kw in str(self.df.iloc[idx]).lower())
                        score = 0.6 + min(0.3, keyword_matches * 0.05)
                        direct_results.append((idx, score))
                
                if direct_results:
                    # Sắp xếp theo điểm số
                    direct_results = sorted(direct_results, key=lambda x: x[1], reverse=True)
                    print(f"Tìm được {len(direct_results)} kết quả trực tiếp cho '{time_unit}'")
                    # Trả về tất cả các kết quả tìm được thay vì chỉ lấy top 3
                    search_results = direct_results[:max_results]
        
        # Nếu không có kết quả, trả về thông báo
        if not search_results:
            suggestions = []
            
            # Đề xuất dựa trên loại truy vấn
            if 'time_focus' in query_info['components']:
                time_col = self.column_names['Định kỳ thực hiện']
                # Lấy các giá trị định kỳ thực tế từ dữ liệu
                unique_periods = [p for p in self.df[time_col].dropna().unique() 
                                 if isinstance(p, str) and p.strip()]
                
                if unique_periods:
                    period_examples = ", ".join(unique_periods[:3])
                    suggestions.append(f"thử tìm với định kỳ cụ thể như: {period_examples}")
            
            return {
                'success': False,
                'error': 'Không tìm thấy thông tin phù hợp',
                'query_info': query_info,
                'suggestions': suggestions,
                'results': []
            }
        
        # Lọc bỏ các kết quả có điểm số quá thấp
        min_threshold = 0.15
        filtered_results = [(idx, score) for idx, score in search_results if score > min_threshold]
        
        # Nếu không còn kết quả sau khi lọc, trả về thông báo
        if not filtered_results:
            return {
                'success': False,
                'error': 'Không tìm thấy thông tin phù hợp hoặc độ tương đồng quá thấp',
                'query_info': query_info,
                'results': []
            }
        
        # Kiểm tra xem có phải là truy vấn về vai trò và giai đoạn không
        is_role_stage_query = False
        role = None
        expected_stages = []
        
        if 'nguoi_thuc_hien' in query_info['components'] and query_info['components']['nguoi_thuc_hien']:
            role = query_info['components']['nguoi_thuc_hien'].lower()
            if role in self.role_stage_mapping:
                expected_stages = self.role_stage_mapping[role]
                is_role_stage_query = any(term in query.lower() for term in ['thuộc', 'giai đoạn', 'làm việc'])
        
        # Kiểm tra xem có phải là truy vấn về thời gian không
        is_time_query = 'time_focus' in query_info['components']
        time_unit = query_info['components'].get('time_unit', '')
        
        # Chuẩn bị kết quả trả về
        results = []
        already_included_stages = set()  # Để tránh trùng lặp giai đoạn
        
        role_col = self.column_names['Người thực hiện']
        stage_col = self.column_names['Giai đoạn']
        time_col = self.column_names['Định kỳ thực hiện']
        
        for idx, score in filtered_results:
            row = self.df.iloc[idx]
            
            # Kiểm tra và loại bỏ các kết quả nếu là tiêu đề hoặc hàng header
            is_header = False
            row_text = ' '.join([str(v) for v in row.values if isinstance(v, str)])
            header_patterns = ['quy trình', 'giai đoạn', 'phòng ban', 'người thực hiện', 'mục tiêu', 'output']
            if sum(1 for pattern in header_patterns if pattern.lower() in row_text.lower()) >= 3:
                is_header = True
            
            # Xử lý truy vấn về vai trò thuộc giai đoạn nào
            if is_role_stage_query and role:
                # Nếu hàng không có vai trò phù hợp, bỏ qua
                if not (isinstance(row[role_col], str) and role in row[role_col].lower()):
                    if score < 0.4:  # Vẫn giữ lại các kết quả có điểm cao
                        continue
                
                # Nếu là truy vấn về giai đoạn, ưu tiên hiển thị các giai đoạn khác nhau
                if isinstance(row[stage_col], str) and row[stage_col].strip():
                    stage = row[stage_col].strip().lower()
                    if stage in already_included_stages and score < 0.5:
                        continue
                    already_included_stages.add(stage)
                
                # Tăng điểm cho các kết quả có giai đoạn dự kiến
                if isinstance(row[stage_col], str) and any(exp_stage.lower() in row[stage_col].lower() for exp_stage in expected_stages):
                    score *= 1.2  # Tăng điểm thêm 20%
            
            # Xử lý truy vấn về thời gian
            if is_time_query and time_unit:
                # Ưu tiên các kết quả có thời gian phù hợp
                if not (isinstance(row[time_col], str) and time_unit.lower() in row[time_col].lower()):
                    if score < 0.5:  # Vẫn giữ lại các kết quả có điểm cao
                        continue
                else:
                    # Tăng điểm cho kết quả phù hợp với đơn vị thời gian
                    score *= 1.3  # Tăng điểm thêm 30%
            
            # Bỏ qua nếu là header hoặc tất cả các cột đều trống
            if is_header or pd.isnull(row).all():
                continue
                
            # Tạo từ điển kết quả sử dụng ánh xạ tên cột thay vì hard-coding
            result = {'score': score}
            
            # Thêm các trường dữ liệu có ý nghĩa, sử dụng tên cột thân thiện
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    # Sử dụng tên thân thiện nếu có, nếu không thì giữ nguyên tên cột
                    friendly_name = self.inverse_column_names.get(col, col)
                    # Chuyển đổi tên trường sang dạng snake_case để dễ sử dụng
                    field_name = friendly_name.lower().replace(' ', '_').replace('/', '_')
                    result[field_name] = value
            
            # Nếu chưa có giai đoạn nhưng có thể suy ra từ các hàng trước
            if 'giai_đoạn' not in result or not result['giai_đoạn'] and idx > 0:
                # Tìm trong các hàng trên để lấy thông tin giai đoạn
                for i in range(idx-1, -1, -1):
                    prev_stage = self.df.iloc[i][stage_col]
                    if isinstance(prev_stage, str) and prev_stage.strip():
                        result['giai_đoạn'] = prev_stage
                        break
            
            # Đảm bảo các trường quan trọng luôn có mặt trong kết quả
            essential_fields = ['giai_đoạn', 'công_việc', 'phòng_ban', 'người_thực_hiện', 'mục_tiêu']
            for field in essential_fields:
                if field not in result:
                    result[field] = ''
            
            results.append(result)
        
        # Nếu là truy vấn về vai trò thuộc giai đoạn nào và không có kết quả cụ thể
        # Trả về các giai đoạn từ bản đồ ánh xạ
        if is_role_stage_query and role and not results:
            if expected_stages:
                for stage in expected_stages:
                    result = {
                        'giai_đoạn': stage,
                        'người_thực_hiện': role.capitalize(),
                        'score': 0.7  # Điểm mặc định cho kết quả từ ánh xạ
                    }
                    results.append(result)
        
        # Nếu là truy vấn về thời gian và không có kết quả cụ thể
        if is_time_query and time_unit and not results:
            # Tìm kiếm trực tiếp một lần nữa với ngưỡng thấp hơn
            direct_time_results = []
            for idx in range(len(self.df)):
                if isinstance(self.df.iloc[idx][time_col], str) and time_unit.lower() in self.df.iloc[idx][time_col].lower():
                    row = self.df.iloc[idx].to_dict()
                    # Chỉ lấy những trường không trống
                    result = {
                        'score': 0.6,
                        'định_kỳ_thực_hiện': self.df.iloc[idx][time_col]
                    }
                    
                    # Thêm các trường quan trọng
                    for field, col_name in [
                        ('giai_đoạn', self.column_names['Giai đoạn']),
                        ('công_việc', self.column_names['Công việc']),
                        ('phòng_ban', self.column_names['Phòng ban']),
                        ('người_thực_hiện', self.column_names['Người thực hiện'])
                    ]:
                        if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip():
                            result[field] = row[col_name]
                        else:
                            result[field] = ''
                    
                    direct_time_results.append(result)
            
            if direct_time_results:
                print(f"Tìm được {len(direct_time_results)} kết quả bổ sung cho '{time_unit}'")
                # Lấy tất cả kết quả, tối đa 10 kết quả
                results = direct_time_results[:max_results]
        
        # Dữ liệu kết quả cuối cùng
        response = {
            'success': True,
            'query_info': query_info,
            'results': self.prepare_results_for_response(query, query_info, filtered_results) if not results else {'results': results}
        }
        
        # Trả về kết quả dưới dạng JSON string hoặc dict tùy theo yêu cầu
        if kwargs.get('return_json', False):
            return json.dumps(response, ensure_ascii=False)
        return response
    
    def extract_role_terms(self, text):
        """Trích xuất các từ khóa về vai trò từ văn bản"""
        # Lấy tất cả vai trò từ cột Người thực hiện
        all_roles = set()
        role_col = self.column_names['Người thực hiện']
        
        for val in self.df[role_col].dropna().unique():
            if isinstance(val, str):
                roles = [r.strip().lower() for r in val.split(',')]
                all_roles.update(roles)
        
        # Kiểm tra từng vai trò có trong văn bản không
        found_roles = []
        for role in all_roles:
            if role in text.lower():
                found_roles.append(role)
        
        return found_roles
    
    def analyze_query(self, query):
        """Phân tích truy vấn và xác định loại truy vấn"""
        import re
        import unidecode
        
        query_info = {
            'original': query,
            'processed': query.lower(),
            'query_type': 'general',
            'search_focus': [],
            'components': {
                'giai_doan': None,
                'phong_ban': None,
                'cong_viec': None,
                'keywords': []
            }
        }
        
        # Phát hiện truy vấn về người/nhân sự
        person_terms = ['ai', 'người', 'nhân viên', 'nhân sự', 'thực hiện', 'phụ trách', 'leader', 'team']
        
        # Phát hiện truy vấn về quy trình
        process_terms = ['giai đoạn', 'quy trình', 'công việc', 'phòng ban', 'bộ phận', 'thuộc']
        
        # Phát hiện truy vấn về mục tiêu/kết quả
        content_terms = ['mục tiêu', 'làm gì', 'kết quả', 'nhiệm vụ', 'trách nhiệm', 'công việc', 'output']
        
        # Phát hiện truy vấn về thời gian và tần suất
        time_terms = ['hàng ngày', 'ngày', 'tuần', 'tháng', 'quý', 'năm', 'định kỳ', 'thường xuyên', 
                     'tần suất', 'khi nào', 'bao lâu', 'thời gian', 'duration', 'always on']
        
        # Đếm số lượng từ khóa theo từng loại
        person_count = sum(1 for term in person_terms if term in query.lower())
        process_count = sum(1 for term in process_terms if term in query.lower())
        content_count = sum(1 for term in content_terms if term in query.lower())
        time_count = sum(1 for term in time_terms if term in query.lower())
        
        # Xác định mục tiêu chính của truy vấn
        if person_count > 0:
            query_info['search_focus'].append('person')
        if process_count > 0:
            query_info['search_focus'].append('process')
        if content_count > 0:
            query_info['search_focus'].append('content')
        if time_count > 0:
            query_info['search_focus'].append('time')
            query_info['components']['time_focus'] = True
            
            # Trích xuất thông tin về đơn vị thời gian cụ thể
            for term in time_terms:
                if term in query.lower():
                    query_info['components']['time_unit'] = term
                    break
        
        # Nếu không xác định được, tìm kiếm tổng quát
        if not query_info['search_focus']:
            query_info['search_focus'] = ['process', 'content', 'person']
        
        # Từ điển biến thể mở rộng
        common_variations = {
            'telesale': 'telesales',
            'tele sale': 'telesales',
            'phone sale': 'telesales',
            'marketing': 'mkt',
            'kinh doanh': 'sales',
            'kd': 'sales', 
            'mkt': 'marketing',
            'tiếp thị': 'marketing',
            'branding': 'branding mkt',
            'data': 'data qualification',
            'nguồn': 'sales sourcing',
            'thiết kế': 'design',
            'tìm kiếm': 'sales sourcing',
            'xác định': 'data qualification',
            'tiếp cận': 'approach',
            'khách hàng': 'customer',
            'đại lý': 'agency',
            'nhân sự': 'hr',
            'nhân lực': 'hr',
            'tuyển dụng': 'hr',
            'hàng tháng': 'tháng',
            'hàng tuần': 'tuần',
            'hàng năm': 'năm',
            'hàng quý': 'quý',
            'định kỳ tháng': 'tháng',
            'định kỳ tuần': 'tuần',
            'định kỳ năm': 'năm'
        }
        
        # Chuẩn hóa query
        processed_query = query.lower()
        for variation, standard in common_variations.items():
            pattern = r'\b' + variation + r'\b' 
            processed_query = re.sub(pattern, standard, processed_query)
        
        query_info['processed'] = processed_query
        
        # Trích xuất vai trò từ truy vấn
        role_terms = self.extract_role_terms(processed_query)
        
        # Xử lý chung cho tất cả vai trò, không chỉ Telesales
        for role in role_terms:
            query_info['components']['nguoi_thuc_hien'] = role.capitalize()
            if 'person' not in query_info['search_focus']:
                query_info['search_focus'].append('person')
            
            # Nếu truy vấn liên quan đến giai đoạn của vai trò
            if any(term in query.lower() for term in ['thuộc', 'giai đoạn', 'làm việc trong']):
                # Xử lý tìm giai đoạn từ bản đồ ánh xạ
                if role in self.role_stage_mapping:
                    if 'stage_keywords' not in query_info['components']:
                        query_info['components']['stage_keywords'] = []
                    
                    for stage in self.role_stage_mapping[role]:
                        query_info['components']['stage_keywords'].append(stage.lower())
                        # Nếu chưa xác định được giai đoạn, lấy giai đoạn đầu tiên
                        if not query_info['components']['giai_doan']:
                            query_info['components']['giai_doan'] = stage
                            
                    # Đảm bảo tìm kiếm theo quy trình
                    if 'process' not in query_info['search_focus']:
                        query_info['search_focus'].append('process')
                    
                    # Thêm các từ khóa giai đoạn vào từ khóa tìm kiếm
                    for stage_kw in query_info['components']['stage_keywords']:
                        words = stage_kw.split()
                        for word in words:
                            if len(word) > 3 and word not in query_info['components']['keywords']:
                                query_info['components']['keywords'].append(word)
        
        # Tìm các giai đoạn trong câu hỏi nếu chưa xác định được
        if not query_info['components']['giai_doan']:
            stages = list(self.df[self.column_names['Giai đoạn']].dropna().unique()) 
            for stage in stages:
                if not isinstance(stage, str):
                    continue
                stage_lower = stage.lower()
                if stage_lower in processed_query or unidecode.unidecode(stage_lower) in unidecode.unidecode(processed_query):
                    query_info['components']['giai_doan'] = stage
                    break
        
        # Tìm các phòng ban trong câu hỏi
        departments = list(self.df[self.column_names['Phòng ban']].dropna().unique())
        for dept in departments:
            if not isinstance(dept, str):
                continue
            dept_lower = dept.lower()
            if dept_lower in processed_query or unidecode.unidecode(dept_lower) in unidecode.unidecode(processed_query):
                query_info['components']['phong_ban'] = dept
                break
        
        # Tìm các thông tin về định kỳ thực hiện
        if 'time_focus' in query_info['components'] and query_info['components']['time_focus']:
            time_col = self.column_names['Định kỳ thực hiện']
            periodic_values = list(self.df[time_col].dropna().unique())
            
            for period in periodic_values:
                if not isinstance(period, str):
                    continue
                period_lower = period.lower()
                if period_lower in processed_query or unidecode.unidecode(period_lower) in unidecode.unidecode(processed_query):
                    query_info['components']['dinh_ky'] = period
                    break
        
        # Tách các từ khóa chính
        for word in processed_query.split():
            if len(word) > 3 and word not in query_info['components']['keywords']:
                query_info['components']['keywords'].append(word)
        
        return query_info
    
    def multi_vector_search(self, query_info, top_n=3):
        """Tìm kiếm đa vector dựa trên nhiều tiêu chí và kết hợp kết quả"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Điều chỉnh top_n dựa vào loại truy vấn
        if 'time_focus' in query_info['components'] and query_info['components']['time_focus']:
            # Tăng số lượng kết quả trả về cho truy vấn thời gian
            top_n = 10  # Tăng lên 10 kết quả thay vì mặc định 3
            print(f"Truy vấn thời gian: Tăng top_n lên {top_n}")
        
        results = []
        scores = []
        
        # Tạo vector truy vấn cho các loại tìm kiếm khác nhau
        query_text = query_info['processed']
        
        # Trọng số cho từng loại tìm kiếm - động dựa trên loại truy vấn
        weights = {
            'title': 0.20,
            'person': 0.15,
            'process': 0.15,
            'content': 0.30,
            'time': 0.10,
            'additional': 0.10
        }
        
        # Điều chỉnh trọng số dựa trên loại truy vấn
        if 'time_focus' in query_info['components'] and query_info['components']['time_focus']:
            # Tăng trọng số thời gian nếu truy vấn liên quan đến thời gian
            weights['time'] = 0.35
            weights['title'] = 0.15
            weights['content'] = 0.20
            weights['person'] = 0.10
            weights['process'] = 0.15
            weights['additional'] = 0.05
        
        # Lấy thông tin về vai trò và giai đoạn dự kiến từ query_info
        role = None
        expected_stages = []
        if 'nguoi_thuc_hien' in query_info['components'] and query_info['components']['nguoi_thuc_hien']:
            role = query_info['components']['nguoi_thuc_hien'].lower()
            # Lấy danh sách giai đoạn dự kiến từ bản đồ ánh xạ
            if role in self.role_stage_mapping:
                expected_stages = self.role_stage_mapping[role]
        
        # Tìm kiếm dựa trên tiêu đề/tên giai đoạn (tìm kiếm tổng hợp)
        if 'process' in query_info['search_focus'] or not query_info['search_focus']:
            title_vector = self.main_vectorizer.transform([query_text])
            title_similarities = cosine_similarity(title_vector, self.main_matrix).flatten()
            
            # Tăng điểm cho các giai đoạn dự kiến của vai trò nếu có
            if expected_stages and any(term in query_text for term in ['thuộc', 'giai đoạn', 'làm việc']):
                for i, index in enumerate(self.df.index):
                    stage_col = self.column_names['Giai đoạn']
                    if isinstance(self.df.loc[index, stage_col], str):
                        for stage in expected_stages:
                            if stage.lower() in self.df.loc[index, stage_col].lower():
                                title_similarities[i] *= 1.5  # Tăng 50% điểm
            
            for i in range(len(self.df)):
                results.append(i)
                scores.append(title_similarities[i] * weights['title'])
        
        # Tìm kiếm dựa trên người thực hiện
        if 'person' in query_info['search_focus']:
            person_vector = self.person_vectorizer.transform([query_text])
            person_similarities = cosine_similarity(person_vector, self.person_matrix).flatten()
            
            # Tăng điểm cho hàng có vai trò phù hợp
            if role:
                role_col = self.column_names['Người thực hiện']
                stage_col = self.column_names['Giai đoạn']
                
                for i, index in enumerate(self.df.index):
                    if isinstance(self.df.loc[index, role_col], str) and role in self.df.loc[index, role_col].lower():
                        person_similarities[i] *= 1.8  # Tăng 80% điểm
                    
                    # Nếu đang tìm kiếm giai đoạn của vai trò
                    if expected_stages and any(term in query_text for term in ['thuộc', 'giai đoạn', 'làm việc']):
                        # Giảm điểm cho các giai đoạn không phù hợp với vai trò
                        if (isinstance(self.df.loc[index, stage_col], str) and 
                            not any(stage.lower() in self.df.loc[index, stage_col].lower() for stage in expected_stages)):
                            person_similarities[i] *= 0.4  # Giảm 60% điểm
            
            for i in range(len(self.df)):
                if i < len(results) and results[i] == i:
                    scores[i] += person_similarities[i] * weights['person']
                else:
                    results.append(i)
                    scores.append(person_similarities[i] * weights['person'])
        
        # Tìm kiếm dựa trên phòng ban & quy trình
        if 'process' in query_info['search_focus']:
            dept_vector = self.process_vectorizer.transform([query_text])
            dept_similarities = cosine_similarity(dept_vector, self.process_matrix).flatten()
            
            for i in range(len(self.df)):
                if i < len(results) and results[i] == i:
                    scores[i] += dept_similarities[i] * weights['process']
                else:
                    results.append(i)
                    scores.append(dept_similarities[i] * weights['process'])
        
        # Tìm kiếm dựa trên nội dung
        if 'content' in query_info['search_focus'] or not query_info['search_focus']:
            content_vector = self.content_vectorizer.transform([query_text])
            content_similarities = cosine_similarity(content_vector, self.content_matrix).flatten()
            
            # Tăng điểm cho các giai đoạn dự kiến của vai trò nếu có
            if role and expected_stages and 'mục tiêu' in query_text:
                role_col = self.column_names['Người thực hiện']
                stage_col = self.column_names['Giai đoạn']
                
                for i, index in enumerate(self.df.index):
                    # Nếu hàng vừa có giai đoạn phù hợp vừa có vai trò phù hợp
                    if (isinstance(self.df.loc[index, stage_col], str) and 
                        any(stage.lower() in self.df.loc[index, stage_col].lower() for stage in expected_stages) and
                        isinstance(self.df.loc[index, role_col], str) and
                        role in self.df.loc[index, role_col].lower()):
                        content_similarities[i] *= 2.0  # Tăng gấp đôi điểm
            
            for i in range(len(self.df)):
                if i < len(results) and results[i] == i:
                    scores[i] += content_similarities[i] * weights['content']
                else:
                    results.append(i)
                    scores.append(content_similarities[i] * weights['content'])
        
        # Tìm kiếm dựa trên thông tin thời gian
        if 'time' in query_info['search_focus'] or 'time_focus' in query_info['components']:
            time_vector = self.time_vectorizer.transform([query_text])
            time_similarities = cosine_similarity(time_vector, self.time_matrix).flatten()
            
            # Tăng điểm cho kết quả phù hợp với đơn vị thời gian cụ thể
            if 'time_unit' in query_info['components'] and query_info['components']['time_unit']:
                time_unit = query_info['components']['time_unit']
                time_col = self.column_names['Định kỳ thực hiện']
                
                for i, index in enumerate(self.df.index):
                    if isinstance(self.df.loc[index, time_col], str) and time_unit.lower() in self.df.loc[index, time_col].lower():
                        time_similarities[i] *= 3.0  # Tăng mạnh điểm số khi khớp đúng đơn vị thời gian
            
            # Tăng điểm cho kết quả phù hợp với định kỳ cụ thể
            if 'dinh_ky' in query_info['components'] and query_info['components']['dinh_ky']:
                specified_period = query_info['components']['dinh_ky']
                time_col = self.column_names['Định kỳ thực hiện']
                
                for i, index in enumerate(self.df.index):
                    if isinstance(self.df.loc[index, time_col], str) and specified_period.lower() in self.df.loc[index, time_col].lower():
                        time_similarities[i] *= 5.0  # Tăng rất mạnh điểm số khi khớp chính xác định kỳ
            
            for i in range(len(self.df)):
                if i < len(results) and results[i] == i:
                    scores[i] += time_similarities[i] * weights['time']
                else:
                    results.append(i)
                    scores.append(time_similarities[i] * weights['time'])
        
        # Tìm kiếm dựa trên thông tin bổ sung
        if 'additional' in query_info['search_focus'] or not query_info['search_focus']:
            additional_vector = self.additional_vectorizer.transform([query_text])
            additional_similarities = cosine_similarity(additional_vector, self.additional_matrix).flatten()
            
            for i in range(len(self.df)):
                if i < len(results) and results[i] == i:
                    scores[i] += additional_similarities[i] * weights['additional']
                else:
                    results.append(i)
                    scores.append(additional_similarities[i] * weights['additional'])
        
        # Tìm kiếm trực tiếp trong từng cột nếu cần
        if 'time_focus' in query_info['components'] or role or 'dinh_ky' in query_info['components']:
            # Danh sách cột cần tìm kiếm trực tiếp
            columns_to_search = []
            
            if 'time_focus' in query_info['components']:
                columns_to_search.append('Định kỳ thực hiện')
                
            if role:
                columns_to_search.append('Người thực hiện')
                
            for col_name in columns_to_search:
                if col_name in self.column_vectorizers:
                    # Tạo vector truy vấn cho cột cụ thể
                    col_vector = self.column_vectorizers[col_name].transform([query_text])
                    col_similarities = cosine_similarity(col_vector, self.column_matrices[col_name]).flatten()
                    
                    # Cập nhật điểm số
                    for i in range(len(self.df)):
                        if i < len(results) and results[i] == i:
                            # Tăng cường điểm số cho tìm kiếm trực tiếp
                            scores[i] += col_similarities[i] * 0.15
                        else:
                            results.append(i)
                            scores.append(col_similarities[i] * 0.15)
        
        # Kết hợp kết quả
        result_dict = {}
        for i, idx in enumerate(results):
            if idx not in result_dict or scores[i] > result_dict[idx]:
                result_dict[idx] = scores[i]
        
        # Sắp xếp kết quả theo điểm số
        sorted_results = sorted([(idx, score) for idx, score in result_dict.items()], 
                                key=lambda x: x[1], reverse=True)
        
        # Xử lý sau cùng cho tìm kiếm theo thời gian
        if 'time_focus' in query_info['components'] and query_info['components']['time_focus']:
            time_col = self.column_names['Định kỳ thực hiện']
            time_unit = query_info['components'].get('time_unit', '')
            
            # Tạo danh sách kết quả phù hợp với đơn vị thời gian
            time_matching_results = []
            other_results = []
            
            for idx, score in sorted_results:
                if (isinstance(self.df.iloc[idx][time_col], str) and 
                    time_unit and time_unit.lower() in self.df.iloc[idx][time_col].lower()):
                    # Tăng điểm cho kết quả phù hợp
                    time_matching_results.append((idx, score * 1.2))
                else:
                    other_results.append((idx, score))
            
            # Kết hợp kết quả, ưu tiên kết quả phù hợp thời gian
            sorted_results = sorted(time_matching_results, key=lambda x: x[1], reverse=True)
            
            # Thêm các kết quả khác nếu cần
            if len(sorted_results) < top_n:
                remaining_slots = top_n - len(sorted_results)
                sorted_results.extend(other_results[:remaining_slots])
            
            print(f"Tìm được {len(time_matching_results)} kết quả phù hợp với đơn vị thời gian '{time_unit}'")
        
        # Loại bỏ các kết quả có điểm số quá thấp
        filtered_results = [(idx, score) for idx, score in sorted_results if score > 0.15]
        
        print(f"Số kết quả sau khi lọc: {len(filtered_results)}, top_n={top_n}")
        
        # Trả về top N kết quả
        return filtered_results[:top_n]
    
    def filter_and_rank_results(self, scores, min_threshold=0.15):
        """Lọc và xếp hạng kết quả dựa trên điểm số"""
        import numpy as np
        
        # Lọc các kết quả có điểm thấp
        valid_indices = np.where(scores > min_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Tính điểm chất lượng dữ liệu cho mỗi hàng
        quality_scores = np.zeros(len(scores))
        
        for idx in valid_indices:
                row = self.df.iloc[idx]
            # Đánh giá chất lượng dữ liệu: có bao nhiêu trường có giá trị
            non_empty_fields = sum(1 for x in row if pd.notna(x) and str(x).strip() != '')
            quality_score = non_empty_fields / len(row)
            
            # Kiểm tra xem dòng này có phải là header không
            # Nếu là header, điểm chất lượng sẽ rất thấp
            if self.is_likely_header_row(row):
                quality_score = 0.01
                
            quality_scores[idx] = quality_score
        
        # Kết hợp điểm số tương đồng và chất lượng
        final_scores = scores * 0.7 + quality_scores * 0.3
        
        # Lấy top 5 kết quả
        top_indices = final_scores.argsort()[-5:][::-1]
        return [(i, final_scores[i]) for i in top_indices if final_scores[i] > min_threshold]
    
    def is_likely_header_row(self, row):
        """Kiểm tra một hàng có khả năng là header không"""
        # Đếm số lượng giá trị có đặc điểm của header
        header_like_count = 0
        value_count = 0
        
        for col, val in row.items():
            if pd.isna(val) or str(val).strip() == '':
                continue
                
            value_count += 1
            val_str = str(val)
            
            # Kiểm tra các đặc điểm của header
            if (val_str.istitle() or val_str.isupper() or 
                val_str in ['A', 'R', 'BOD', 'CEO'] or
                val_str in ['Giai đoạn', 'Phòng ban', 'Mục tiêu']):
                header_like_count += 1
        
        # Nếu >60% giá trị có đặc điểm của header
        return value_count > 0 and header_like_count / value_count > 0.6
    
    def is_likely_header_value(self, value):
        """Kiểm tra một giá trị có khả năng là giá trị header không"""
        if not isinstance(value, str):
            return False
            
        # Các chuỗi header điển hình
        header_values = ['Giai đoạn', 'Công việc', 'Phòng ban', 'Mục tiêu',
                         'A', 'R', 'Làm gì', 'Kết quả trả ra', 'Duration']
        
        if value in header_values:
            return True
            
        # Kiểm tra nếu viết hoa và độ dài ngắn
        if value.isupper() and len(value) < 15:
            return True
            
        # Kiểm tra Title Case cho các từ ngắn
        if value.istitle() and len(value.split()) <= 3:
            return True
            
        return False
    
    def prepare_results_for_response(self, original_query, query_info, ranked_indices_scores):
        """Chuẩn bị kết quả cuối cùng cho phản hồi"""
        import json
        
        if not ranked_indices_scores:
            suggestions = []
            
            # Đề xuất giai đoạn
            if not query_info['components']['giai_doan']:
                stages = [str(s) for s in self.df[self.column_names['Giai đoạn']].dropna().unique() if isinstance(s, str) and str(s).strip()]
                if stages:
                    sample_stages = ", ".join(stages[:3]) + "..."
                    suggestions.append(f"thêm giai đoạn cụ thể (ví dụ: {sample_stages})")
            
            # Đề xuất phòng ban
            if not query_info['components']['phong_ban']:
                depts = [str(d) for d in self.df[self.column_names['Phòng ban']].dropna().unique() if isinstance(d, str) and str(d).strip()]
                if depts:
                    sample_depts = ", ".join(depts[:3]) + "..."
                    suggestions.append(f"thêm phòng ban cụ thể (ví dụ: {sample_depts})")
            
            # Đề xuất thời gian nếu đang tìm kiếm theo thời gian
            if 'time_focus' in query_info['components']:
                time_col = self.column_names['Định kỳ thực hiện']
                periods = [str(p) for p in self.df[time_col].dropna().unique() if isinstance(p, str) and str(p).strip()]
                if periods:
                    sample_periods = ", ".join(periods[:3]) + "..."
                    suggestions.append(f"thêm định kỳ thực hiện cụ thể (ví dụ: {sample_periods})")
            
            suggest_text = ""
            if suggestions:
                suggest_text = f"Bạn có thể thử {' hoặc '.join(suggestions)}."
            
            return {
                "query_info": query_info,
                "message": f"Không tìm thấy thông tin phù hợp với câu hỏi của bạn. {suggest_text}",
                "total_results": 0,
                "results": []
            }
        
        results = []
        for idx, score in ranked_indices_scores:
            row = self.df.iloc[idx]
            
            # Kiểm tra xem hàng có phải là header/tiêu đề hay không
            if self.is_likely_header_row(row):
                continue
            
            # Đánh giá độ phù hợp
            relevance = "Cao" if score > 0.5 else "Trung bình" if score > 0.3 else "Thấp"
            
            # Chuyển đổi row thành dictionary có cấu trúc
                result = {
                "do_phu_hop": relevance,
                "diem_so": round(score, 2)
            }
            
            # Thêm các trường dữ liệu có ý nghĩa
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    # Sử dụng tên thân thiện nếu có, nếu không thì giữ nguyên tên cột
                    friendly_name = self.inverse_column_names.get(col, col)
                    # Kiểm tra nếu giá trị là header/tiêu đề
                    if not self.is_likely_header_value(val):
                        result[friendly_name.lower().replace(' ', '_')] = val
            
            # Kiểm tra nếu đây là truy vấn liên quan đến thời gian, đảm bảo trường thời gian được bao gồm
            if 'time_focus' in query_info['components']:
                time_col = self.column_names['Định kỳ thực hiện']
                time_val = row[time_col]
                if pd.notna(time_val) and str(time_val).strip():
                    result['định_kỳ_thực_hiện'] = time_val
                    
                    # Tăng thêm điểm nếu thời gian khớp với yêu cầu
                    if 'time_unit' in query_info['components']:
                        time_unit = query_info['components']['time_unit']
                        if time_unit and time_unit.lower() in str(time_val).lower():
                            # Điều chỉnh độ phù hợp
                            result['do_phu_hop'] = "Cao"
                            result['diem_so'] = round(min(0.95, score * 1.3), 2)
            
                results.append(result)
        
        # Sắp xếp kết quả theo độ phù hợp nếu tìm theo thời gian
        if 'time_focus' in query_info['components'] and 'time_unit' in query_info['components']:
            time_unit = query_info['components']['time_unit']
            
            # Sắp xếp để ưu tiên các kết quả có thời gian khớp
            def time_sort_key(result):
                # Nếu có trường định_kỳ_thực_hiện và nó chứa time_unit
                if 'định_kỳ_thực_hiện' in result and time_unit in str(result['định_kỳ_thực_hiện']).lower():
                    return (1, result['diem_so'])  # Ưu tiên cao nhất
                return (0, result['diem_so'])      # Ưu tiên thấp hơn
                
            results = sorted(results, key=time_sort_key, reverse=True)
        
        return {
            "query_info": query_info,
            "total_results": len(results),
            "results": results
        }

# ✅ Chức năng lưu và nạp lịch sử hội thoại
def save_conversation(messages, conversation_name=None, path="./history"):
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Tạo tên file từ thời gian hoặc tên được chỉ định
    if conversation_name:
        # Làm sạch tên file
        conversation_name = ''.join(c for c in conversation_name if c.isalnum() or c in ' -_')
        filename = f"{path}/{conversation_name}.json"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{path}/chat_{timestamp}.json"
    
    # Lưu hội thoại vào file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    
    return filename

def load_conversation(filename_or_index, path="./history"):
    # Kiểm tra nếu thư mục tồn tại
    if not os.path.exists(path):
        return None, "Chưa có lịch sử hội thoại nào được lưu."
    
    # Lấy danh sách file
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    
    if not files:
        return None, "Không tìm thấy file lịch sử hội thoại nào."
    
    # Sắp xếp theo thời gian tạo (mới nhất trước)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
    
    filename = None
    # Nếu là số, lấy file theo index
    if str(filename_or_index).isdigit():
        index = int(filename_or_index)
        if 0 <= index < len(files):
            filename = os.path.join(path, files[index])
        else:
            return None, f"Chỉ số không hợp lệ. Vui lòng chọn từ 0 đến {len(files)-1}."
    else:
        # Nếu là tên file, tìm file phù hợp
        for file in files:
            if filename_or_index in file:
                filename = os.path.join(path, file)
                break
    
    if not filename:
        return None, f"Không tìm thấy file '{filename_or_index}'. Sử dụng '/list' để xem danh sách hội thoại."
    
    # Đọc file
    try:
        with open(filename, "r", encoding="utf-8") as f:
            messages = json.load(f)
        return messages, f"Đã tải lịch sử hội thoại từ {filename}"
    except Exception as e:
        return None, f"Lỗi khi đọc file: {str(e)}"

def list_conversations(path="./history"):
    if not os.path.exists(path):
        return "Chưa có lịch sử hội thoại nào được lưu."
    
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    
    if not files:
        return "Không tìm thấy file lịch sử hội thoại nào."
    
    # Sắp xếp theo thời gian tạo (mới nhất trước)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
    
    result = "Danh sách hội thoại đã lưu:\n"
    for i, file in enumerate(files):
        # Lấy thời gian tạo
        timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(path, file)))
        result += f"{i}: {file} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    result += "\nSử dụng '/load [số thứ tự]' hoặc '/load [tên file]' để tải lịch sử."
    return result

# ✅ Step 2: Cấu hình LLM
llm_cfg = {
    'model': 'qwen3-30b-a3b',
    'model_server': 'http://localhost:1234/v1',  # LM Studio mặc định
    'api_key': 'EMPTY',
}

# ✅ Step 3: Tạo Assistant
assistant = Assistant(
    llm=llm_cfg,
    function_list=["tra_cuu_quy_trinh"],
    system_message="""Bạn là chuyên gia về mô hình RACI của công ty, giúp trả lời câu hỏi về quy trình công việc dựa trên mô hình RACI.

Nguyên tắc làm việc:
1. Luôn gọi tool tra_cuu_quy_trinh khi trả lời câu hỏi về quy trình công việc
2. Nếu nhận được dữ liệu JSON, hãy biến đổi nó thành văn bản có cấu trúc rõ ràng, dễ đọc
3. Nếu nhiều kết quả, hãy tóm tắt điểm chung và nêu rõ sự khác biệt
4. Nếu không tìm thấy thông tin, hãy gợi ý người dùng cung cấp thêm chi tiết về giai đoạn, phòng ban, hoặc công việc cụ thể
5. Trả lời ngắn gọn, súc tích, có cấu trúc rõ ràng
6. Nếu không có dữ liệu từ tool, thì hãy nói rõ mình không tìm thấy dữ liệu và muốn trả lời theo kiến thức của mô hình không.

Khi cần biết thêm thông tin, hãy hỏi về:
- Giai đoạn quy trình: Branding MKT, Sales Sourcing, Data Qualification, Approach, v.v.
- Phòng ban: Marketing, Kinh doanh, v.v.
- Loại công việc cụ thể
"""
)

# ✅ Step 4: Giao diện chat đơn giản
messages = []
max_history = 6  # 3 cặp hội thoại (3 user + 3 assistant)

# Hiển thị thông tin chào mừng
print("\n" + "="*50)
print("🤖 CHATBOT QUY TRÌNH CÔNG VIỆC")
print("="*50)
print("Các lệnh đặc biệt:")
print("/save [tên] - Lưu hội thoại hiện tại")
print("/load [số|tên] - Tải lịch sử hội thoại")
print("/list - Xem danh sách hội thoại đã lưu")
print("/clear - Xóa lịch sử hội thoại hiện tại")
print("/exit - Thoát chương trình")
print("="*50 + "\n")

while True:
    query = input("❓ Câu hỏi: ").strip()
    if not query:
        continue
        
    # Xử lý các lệnh đặc biệt
    if query.startswith("/"):
        cmd_parts = query.split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        
        # Lệnh thoát
        if cmd == "/exit":
            print("👋 Tạm biệt! Hẹn gặp lại.")
            break
            
        # Lệnh lưu hội thoại
        elif cmd == "/save":
            name = cmd_parts[1] if len(cmd_parts) > 1 else None
            saved_file = save_conversation(messages, name)
            print(f"🤖 Đã lưu hội thoại vào {saved_file}")
            continue
            
        # Lệnh tải hội thoại
        elif cmd == "/load":
            if len(cmd_parts) < 2:
                print("🤖 Vui lòng cung cấp số thứ tự hoặc tên file. Sử dụng /list để xem danh sách.")
            else:
                loaded_messages, message = load_conversation(cmd_parts[1])
                if loaded_messages:
                    messages = loaded_messages
                    print(f"🤖 {message}")
                else:
                    print(f"🤖 {message}")
            continue
            
        # Lệnh liệt kê hội thoại
        elif cmd == "/list":
            conversations_list = list_conversations()
            print(f"🤖 {conversations_list}")
            continue
            
        # Lệnh xóa lịch sử
        elif cmd == "/clear":
            messages = []
            print("🤖 Đã xóa lịch sử hội thoại hiện tại.")
            continue
    
    # Xử lý câu hỏi bình thường
    messages.append({'role': 'user', 'content': query})

    # Hiển thị đang xử lý
    loading_chars = "|/-\\"
    for i in range(5):
        print(f"\r🤖 Đang suy nghĩ {loading_chars[i % len(loading_chars)]}", end='', flush=True)
        time.sleep(0.2)
    print("\r🤖 Trả lời: ", end='', flush=True)
    
    # Gọi trợ lý và xử lý phản hồi
    response = []
    response_text = ''
    for r in assistant.run(messages=messages):
        chunk = r[-1]['content']
        print(chunk, end='', flush=True)
        response_text += chunk
        response.append(r[-1])
    messages.extend(response)
    
    # Giữ lại lịch sử theo cặp hội thoại
    if len(messages) > max_history:
        # Chỉ giữ lại các cặp hoàn chỉnh, bắt đầu với user
        # Tìm các cặp hoàn chỉnh từ cuối lên
        new_messages = []
        pairs_count = 0
        i = len(messages) - 2  # Bắt đầu từ cặp cuối cùng
        
        while i >= 0 and pairs_count < (max_history // 2):
            if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                new_messages = [messages[i], messages[i+1]] + new_messages
                pairs_count += 1
                i -= 2
            else:
                i -= 1
                
        # Nếu tin nhắn cuối là user và chưa được trả lời
        if messages and messages[-1]['role'] == 'user' and messages[-1] not in new_messages:
            new_messages.append(messages[-1])
            
        messages = new_messages
    
    print()  # xuống dòng sau trả lời