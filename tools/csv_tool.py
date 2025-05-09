import json
import pandas as pd
from qwen_agent.tools.base import BaseTool, register_tool

df = pd.read_csv("nhan_vien.csv")

@register_tool("tra_cuu_nhan_vien")
class CSVTool(BaseTool):
    description = "Tra cứu thông tin nhân viên từ file CSV"
    parameters = [{
        "name": "query",
        "type": "string",
        "description": "Câu hỏi như: 'Lương của An là bao nhiêu?'",
        "required": True
    }]

    def call(self, params: str, **kwargs) -> str:
        q = json.loads(params)["query"].lower()
        for _, row in df.iterrows():
            if row['Ten'].lower() in q:
                return f"{row['Ten']} thuộc phòng {row['PhongBan']}, lương là {row['Luong']}."
        return "Không tìm thấy nhân viên phù hợp."
