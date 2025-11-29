from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "./data/viquad_raw"
    output_dir: str = "./checkpoints/qwen_viquad_final"
    max_seq_length: int = 2048

    # ====== BATCH SIZE TỐI ƯU 32GB VRAM ======
    per_device_train_batch_size: int = 24   # RTX 5090 chạy thoải mái
    per_device_eval_batch_size: int = 24
    gradient_accumulation_steps: int = 1    # Không cần accumulate nữa

    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05

    # ====== LoRA ======
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # ====== Hardware ======
    bf16: bool = True
    gradient_checkpointing: bool = False  # TẮT để tốc độ x3
    seed: int = 42