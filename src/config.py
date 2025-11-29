from dataclasses import dataclass, field
from typing import List, Optional


from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "./data/viquad_raw"
    output_dir: str = "./checkpoints/qwen_viquad_final"
    max_seq_length: int = 1536   # nếu vẫn OOM nữa thì hạ xuống 1536 hoặc 1024

    # ====== BATCH SIZE AN TOÀN CHO 32GB ======
    per_device_train_batch_size: int = 8      # ↓ từ 16/24 xuống 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2      # Effective batch = 8*4 = 32

    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Giảm bớt module cho LoRA để đỡ tốn VRAM
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"   # bỏ gate/up/down cho nhẹ
    ])

    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

    # BẬT LẠI checkpointing để tiết kiệm VRAM
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
