import json
import os
import tiktoken
import re
import torch
import numpy as np
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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

def semantic_chunk_documents(
    docs,
    model_path,
    max_token_size=512,
    overlap_token_size=64,
    batch_size=32,
):
    """
    使用语义相似度进行文档切分
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model from {model_path} on {device}...")
    
    # model_kwargs = {'trust_remote_code': True, 'device': device} # Replaced with SentenceTransformer
    try:
        model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"Failed to load sentence transformer model: {e}")
        raise e
        
    ENCODER = tiktoken.get_encoding("cl100k_base")
    results = []
    
    for doc_text in docs:
        # 优先处理带有引号的句子结束符，然后再处理普通结束符
        # pattern: 标点符号 + 可选的引号 + 可选的空白
        # 使用特殊分隔符进行切分，避免误伤原有的换行符(虽然原逻辑也不切分无标点换行)
        # Pattern matches: [.!?。！？] followed optionally by [”」], then whitespace
        doc_text = re.sub(r'([.!?。！？][”」]?)\s*', r'\1<|SEP|>', doc_text)
        sentences = doc_text.split('<|SEP|>')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
            
        # 使用 batch_size 进行批量推理，并显示进度条
        print(f"Encoding {len(sentences)} sentences...")
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True # Optional: normalize for cosine similarity
        )
        
        sentence_lens = [len(ENCODER.encode(s)) for s in sentences]
        
        similarities = []
        for i in range(len(sentences) - 1):
            vec_a = embeddings[i]
            vec_b = embeddings[i+1]
            
            # Since we used normalize_embeddings=True, we can just use dot product
            # But to be safe and consistent with previous logic (handling zeros), we'll keep the safe division check or simplified dot product
            # If normalized, norms are 1.
            
            # Replicating logic using dot product which represents cosine similarity for normalized vectors
            sim = np.dot(vec_a, vec_b) 
            similarities.append(sim)
            
        current_chunk_indices = []
        current_chunk_len = 0
        
        i = 0
        while i < len(sentences):
            current_chunk_indices.append(i)
            current_chunk_len += sentence_lens[i]
            
            # Check if we should split
            # 1. Hard limit check (avoid infinite growth) or End of Doc
            is_last = (i == len(sentences) - 1)
            
            if current_chunk_len >= max_token_size or is_last:
                # Decide where to split
                split_at_idx = -1 # Index within current_chunk_indices
                
                if current_chunk_len > max_token_size:
                     # We exceeded. Look for best semantic split in the allowed window
                     # Window: [max - overlap, max]
                     
                     best_sim = 1.0
                     best_j = -1
                     
                     temp_len = 0
                     for j, idx_val in enumerate(current_chunk_indices):
                         temp_len += sentence_lens[idx_val]
                         
                         # Check if this boundary is a candidate
                         # Boundary is *after* sentences[idx_val].
                         if temp_len >= (max_token_size - overlap_token_size) and temp_len <= max_token_size:
                             if idx_val < len(similarities):
                                 sim = similarities[idx_val]
                                 if sim < best_sim:
                                     best_sim = sim
                                     best_j = j
                     
                     if best_j != -1:
                         split_at_idx = best_j
                     else:
                         # No semantic split found. 
                         # Default: Split before the sentence that broke the limit (if possible)
                         if len(current_chunk_indices) > 1:
                             split_at_idx = len(current_chunk_indices) - 2 
                         else:
                             split_at_idx = 0 
                
                else: 
                     # Reached end of doc with valid size
                     split_at_idx = len(current_chunk_indices) - 1
                
                # Perform Split
                chunk_indices = current_chunk_indices[:split_at_idx+1]
                chunk_text = "".join([sentences[idx] for idx in chunk_indices])
                
                results.append({
                    "hash_code": compute_mdhash_id(chunk_text),
                    "text": chunk_text
                })
                
                # If we processed the last sentence, we are done
                last_included_idx = chunk_indices[-1]
                if last_included_idx == len(sentences) - 1:
                    break
                
                # Determine overlap for next chunk
                overlap_tokens_count = 0
                overlap_start_idx = last_included_idx + 1 # Default: no overlap
                
                for back_i in range(last_included_idx, -1, -1):
                     overlap_tokens_count += sentence_lens[back_i]
                     if overlap_tokens_count > overlap_token_size:
                         overlap_start_idx = back_i 
                         break
                     overlap_start_idx = back_i 
                     
                # Ensure progress: Start index must be > current_chunk_indices[0]
                if overlap_start_idx <= current_chunk_indices[0]:
                    overlap_start_idx = current_chunk_indices[0] + 1
                    
                i = overlap_start_idx
                current_chunk_indices = []
                current_chunk_len = 0
                continue 
            
            i += 1
            
    
    # Cleanup memory
    del model
    if 'embeddings' in locals():
        del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

if __name__ == "__main__":
    max_token_size = 2048
    overlap_token_size = 128
    
    # Configuration
    use_semantic = True
    # embedding_model_path = "E:/AI/LLM/Models/Embedding/Youtu-Embedding" 
    # Use raw string or forward slashes to avoid escape issues
    embedding_model_path = r"/Models/Embedding/Youtu-Embedding"

    
    # 示例输入路径，用户可修改此变量
    # original_text_file = "datasets/mix/mix.jsonl" 
    original_text_file = "./input/前桌的李同学_校对.txt" # 示例: 纯文本输入

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
            if use_semantic:
                print(f"Using semantic chunking with model: {embedding_model_path}")
                try:
                    results = semantic_chunk_documents(
                        data,
                        model_path=embedding_model_path,
                        max_token_size=max_token_size,
                        overlap_token_size=overlap_token_size,
                    )
                except Exception as e:
                    print(f"Error during semantic chunking: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to standard chunking...")
                    results = chunk_documents(
                        data,
                        model_name="cl100k_base",
                        max_token_size=max_token_size,
                        overlap_token_size=overlap_token_size,
                    )
            else:
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