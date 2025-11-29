import torch
import pandas as pd
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .config import TrainConfig
from .data import load_viquad, extract_answer, format_chat_prompt


def generate_response(model, tokenizer, context, question, max_new_tokens=256):
    """Hàm sinh câu trả lời từ model"""
    # 1. Tạo prompt (Chỉ gồm Context + Question)
    messages = format_chat_prompt(context, question, answer=None)

    # 2. Tokenize & Add Generation Prompt
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 3. Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Có thể chỉnh thành False để Greedy Search (ổn định hơn)
            temperature=0.3,  # Thấp để bớt "ảo giác"
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. Decode (Cắt bỏ phần Prompt input, chỉ lấy phần Model trả lời)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def main():
    cfg = TrainConfig()

    print(f" Loading Base Model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
    )

    # Load LoRA Adapter
    print(f"🔗 Loading LoRA Adapter from: {cfg.output_dir}")
    try:
        model = PeftModel.from_pretrained(base_model, cfg.output_dir)
        print("✅ Adapter loaded successfully!")
    except Exception as e:
        print(f"⚠️ Không load được Adapter (có thể chưa train xong?). Lỗi: {e}")
        print("Running with Base Model only...")
        model = base_model

    model.eval()  # Chuyển sang chế độ eval

    # Load Metrics
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    # Load Data
    print(" Loading Test Data...")
    raw_ds = load_viquad(cfg.data_path)
    test_data = raw_ds["test"]

    # --- CẤU HÌNH TEST ---
    # Để test nhanh thì lấy 20 mẫu, muốn test hết thì bỏ dòng dưới đi
    # test_samples = test_data.select(range(20)) 
    test_samples = test_data  # Chạy full test set (sẽ lâu)

    results = []
    references_bleu = []
    predictions = []

    print(f" Starting Generation on {len(test_samples)} samples...")

    for i, sample in enumerate(tqdm(test_samples)):
        context = sample["context"]
        question = sample["question"]
        ground_truth = extract_answer(sample)

        # Generate
        pred = generate_response(model, tokenizer, context, question)

        # Lưu kết quả để tính điểm
        predictions.append(pred)
        # BLEU yêu cầu list of list references [[ref1, ref2], ...]
        references_bleu.append([ground_truth])

        results.append({
            "id": sample.get("id", i),
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred
        })

    # Tính điểm
    print(" Computing Metrics...")
    bleu_score = bleu.compute(predictions=predictions, references=references_bleu)
    rouge_score = rouge.compute(predictions=predictions, references=references_bleu)

    print("\n" + "=" * 30)
    print(f" BLEU Score: {bleu_score['score']:.2f}")
    print(f" ROUGE-1: {rouge_score['rouge1']:.4f}")
    print(f" ROUGE-2: {rouge_score['rouge2']:.4f}")
    print(f" ROUGE-L: {rouge_score['rougeL']:.4f}")
    print("=" * 30 + "\n")

    # Lưu file Excel/CSV
    df = pd.DataFrame(results)
    save_path = f"{cfg.output_dir}/eval_results.csv"
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f" Detailed results saved to: {save_path}")


if __name__ == "__main__":
    main()