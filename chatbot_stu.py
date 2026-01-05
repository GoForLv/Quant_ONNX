import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os
import time

# model_path = "export/qwen3_fp32.onnx"
model_path = "quant/qwen3_int8.onnx"
tokenizer_path = "./Qwen3-1.7B"

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# ================= TODO 6: 实现自回归生成循环 =================
def generate(prompt, max_tokens=500):
    # 1. 预处理 Prompt
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    
    print(f"Qwen: ", end="", flush=True)
    
    start_time = time.time()
    for i in range(max_tokens):
        # [YOUR CODE HERE]
        # 1. 构造推理输入字典 ort_inputs
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=np.int64)
        }

        # 2. 执行推理 sess.run
        # outputs = ...
        outputs = sess.run(None, ort_inputs)

        # 3. 获取下一个 token 的 ID (提示：取 logits 的最后一个位置，做 argmax)
        # next_token = ...
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
        next_token = np.argmax(next_token_logits)

        # 4. 结束条件判断 (EOS token)
        # if next_token == tokenizer.eos_token_id: break
        if next_token == tokenizer.eos_token_id:
            break

        # 5. 打印当前生成的字
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)
        
        # 6. 更新 input_ids (将新 token 拼接到末尾)
        # input_ids = np.append(...)
        next_token_arr = np.array([[next_token]], dtype=np.int64)
        input_ids = np.append(input_ids, next_token_arr, axis=1)
        
        # pass # 删除此行

        if i % 20 == 0:
            token_per_sec = 20 / (time.time() - start_time)
            print(f"({token_per_sec:.3f}tps)", end="")
            start_time = time.time()

    print("\n")

while True:
    q = input("\nUser: ")
    if q == "exit": break
    generate(q)