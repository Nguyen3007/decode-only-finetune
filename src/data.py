import os
from typing import Dict, Any, List
from datasets import load_from_disk, load_dataset, DatasetDict


# =======================
# LOAD DATASET
# =======================
def load_viquad(data_path: str) -> DatasetDict:
    """Load dataset ViQuAD dạng Arrow hoặc tải online."""
    if os.path.exists(data_path):
        print(f"📂 Loading dataset from disk: {data_path}")
        return load_from_disk(data_path)
    else:
        print("⚠️ Dataset folder not found. Downloading online...")
        return load_dataset("taidng/UIT-ViQuAD2.0")


# =======================
# ANSWER EXTRACTION
# =======================
def extract_answer(example: Dict[str, Any]) -> str:
    # --- FIX LỖI NONETYPE ---
    # Lấy object ra, nếu nó là None thì gán bằng dict rỗng {}
    answers_obj = example.get("answers")
    if answers_obj is None:
        answers_obj = {}

    plausible_obj = example.get("plausible_answers")
    if plausible_obj is None:
        plausible_obj = {}

    # Giờ thì an toàn để .get("text")
    answers = answers_obj.get("text", [])
    plausible = plausible_obj.get("text", [])

    # 1. Ưu tiên câu trả lời chính xác
    if answers and len(answers) > 0 and answers[0]:
        return answers[0]

    # 2. Nếu không, lấy câu trả lời khả dĩ
    if plausible and len(plausible) > 0 and plausible[0]:
        return plausible[0]

    # 3. Trả lời mặc định
    return "Thông tin này không có trong đoạn văn được cung cấp."


# =======================
# CHAT TEMPLATE FORMATTER
# =======================
def format_chat_prompt(context: str, question: str, answer: str = None) -> List[Dict]:
    """
    Format hội thoại theo chuẩn Qwen ChatML:
    [
        {"role": "system", "content": ...}
        {"role": "user", "content": ...}
        {"role": "assistant", "content": answer}  # khi training
    ]
    """

    user_content = (
        "Dựa vào văn bản sau, hãy trả lời câu hỏi một cách ngắn gọn và chính xác.\n\n"
        f"📄 VĂN BẢN:\n{context}\n\n"
        f"❓ CÂU HỎI:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý AI hữu ích. "
                "Hãy trả lời dựa đúng thông tin trong đoạn văn. "
                "Nếu không có thông tin, hãy nói rõ rằng không có dữ liệu."
            )
        },
        {"role": "user", "content": user_content},
    ]

    # Khi training → đưa answer vào để model học
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    return messages
