from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # ====== MODEL & DATA ======
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "./data/viquad_raw"
    output_dir: str = "./checkpoints/qwen_viquad_final"
    max_seq_length: int = 1024

    # ====== TRAINING HYPERPARAMS (RTX 3090 OPTIMIZED) ======
    num_train_epochs: int = 1

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8

    gradient_accumulation_steps: int = 4

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05

    # ====== LORA CONFIG ======
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # ====== LOGGING & SAVING ======
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

    # ====== HARDWARE & OPTIMIZATION ======
    bf16: bool = True

    #  TẮT CHECKPOINTING ĐỂ MAX TỐC ĐỘ (Vì Batch 3 đủ nhỏ để không OOM)
    # Nếu vẫn bị OOM, hãy sửa thành True
    gradient_checkpointing: bool = False

    seed: int = 42