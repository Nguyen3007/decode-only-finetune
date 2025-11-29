import os
import shutil
from datasets import load_dataset, load_from_disk

# Cấu hình
DATASET_NAME = "taidng/UIT-ViQuAD2.0"
SAVE_PATH = "viquad_raw"  # Nơi lưu dataset trên máy bạn


def download_and_inspect():
    print(f" Bắt đầu tải dataset: {DATASET_NAME}...")

    # 1. Tải từ Hugging Face Hub
    # Nó sẽ tự cache vào ~/.cache/huggingface/datasets, nhưng ta sẽ lưu riêng
    ds = load_dataset(DATASET_NAME)

    print("\n Tải xong! Thông tin dataset:")
    print(ds)

    # 2. In thử mẫu dữ liệu để soi structure
    print("\n Soi mẫu đầu tiên của tập Train:")
    first_sample = ds['train'][0]
    for key, value in first_sample.items():
        print(f"   - {key}: {value}")

    # 3. Lưu xuống đĩa (Local Disk)
    # Việc này giúp các file sau này (dec_data.py) load cực nhanh, không cần mạng
    if os.path.exists(SAVE_PATH):
        print(f"\n⚠ Thư mục {SAVE_PATH} đã tồn tại. Đang xóa để lưu mới...")
        shutil.rmtree(SAVE_PATH)

    print(f"\n Đang lưu dataset xuống: {SAVE_PATH} ...")
    ds.save_to_disk(SAVE_PATH)

    print(" Hoàn tất! Bạn có thể kiểm tra thư mục 'data/viquad_raw'")


def verify_load_local():
    """Hàm test thử load lại từ đĩa xem có lỗi không"""
    print("\n Kiểm tra: Load lại từ ổ cứng...")
    try:
        ds_local = load_from_disk(SAVE_PATH)
        print(f" Load thành công. Số lượng train samples: {len(ds_local['train'])}")
    except Exception as e:
        print(f" Lỗi khi load local: {e}")


if __name__ == "__main__":
    download_and_inspect()
    verify_load_local()