import json
import os
import tiktoken
from hashlib import md5
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token_ids = []
        lengths = []

        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk = tokens[start : start + max_token_size]
            chunk_token_ids.append(chunk)
            lengths.append(len(chunk))

        # 解码所有 chunk
        chunk_texts = ENCODER.decode_batch(chunk_token_ids)

        for i, text in enumerate(chunk_texts):
            results.append({
                # "tokens": lengths[i],
                "hash_code": compute_mdhash_id(text), ##使用hash进行编码
                "text": text.strip().replace("\n", ""),
                # "chunk_order_index": i,
            })

    return results

if __name__ == "__main__":
    max_token_size = 1024
    overlap_token_size = 128
    
    # 示例输入路径，用户可修改此变量
    # original_text_file = "datasets/mix/mix.jsonl" 
    original_text_file = "D:/custom/path/input.txt" # 示例: 纯文本输入

    # 获取完整路径
    original_text_file = os.path.abspath(original_text_file)
    
    # 提取文件名(不含后缀)
    file_name_with_ext = os.path.basename(original_text_file)
    file_name_no_ext, ext = os.path.splitext(file_name_with_ext)
    
    # 构建输出目录: workdir/<filename_no_ext>
    output_dir = os.path.join(os.getcwd(), 'workdir', file_name_no_ext)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    chunk_text_file = os.path.join(output_dir, f"{file_name_no_ext}_chunk.json")
    
    data = []
    
    print(f"Processing file: {original_text_file}")
    
    if not os.path.exists(original_text_file):
        print(f"Error: File not found: {original_text_file}")
        # 为了演示，如果文件不存在，我们创建一个测试文件
        if not os.path.exists(original_text_file):
             # 仅用于本地测试流程，实际使用中应报错或处理
             pass
    else:
        if ext.lower() == '.jsonl':
            with open(original_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if 'input' in item and 'context' in item: # Adapt to mix.jsonl structure or generic
                             data.append(item['context']) # Assuming context is the text to chunk
                        elif 'text' in item:
                             data.append(item['text'])
        elif ext.lower() == '.txt':
            with open(original_text_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    data.append(content)
        
        if data:
            results = chunk_documents(
                data,
                max_token_size=max_token_size,
                overlap_token_size=overlap_token_size,
            )
            
            with open(chunk_text_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"Chunking complete. Output saved to: {chunk_text_file}")
        else:
            print("No data found to process.")