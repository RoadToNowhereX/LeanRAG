import json
import os
import tiktoken
import re
import torch
import numpy as np
import gc
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
from hashlib import md5
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def split_text_by_sentences(text, max_sentence_length=1024):
    """
    Split text into sentences, ignoring delimiters inside quotes (both EN "" and CN 「」).
    Supports: . ! ? ... … and their variants ！？。
    Note: EN "" is dangerous! it does not distinguish start and end, only correct grammar can be handled.
    params:
        max_sentence_length: If a sentence (or the current accumulating text) exceeds this length,
                             we assume quotes might be unclosed or too long, and force checking for delimiters.
    """
    sentences = []
    current_start = 0
    
    # Map opening quotes to closing quotes
    quote_pairs = {'"': '"', '“': '”', '「': '」'}
    
    in_quote = False
    expected_close_quote = None
    
    i = 0
    length = len(text)
    
    while i < length:
        char = text[i]
        
        # Check current length from start
        current_len = i - current_start
        
        # If we exceed max_sentence_length, we force in_quote to False to potentially break the sentence
        if in_quote and current_len > max_sentence_length:
            in_quote = False
            expected_close_quote = None
        
        # 1. Handle Quotes
        if in_quote:
            if char == expected_close_quote:
                # Closing the current quote
                in_quote = False
                expected_close_quote = None
        else:
            # Check for opening quotes
            if char in quote_pairs:
                in_quote = True
                expected_close_quote = quote_pairs[char]
                
        # 2. Check for Delimiter if NOT in quote
        if not in_quote:
            # Check if we are at a delimiter
            match_len = 0
            
            # Check for ...
            if i + 3 <= length and text[i:i+3] == '...':
                match_len = 3
            elif char == '…':
                match_len = 1
            elif char in '.!?。！？':
                match_len = 1
            
            if match_len > 0:
                 # Found a delimiter outside quotes
                 end_idx = i + match_len
                 
                 # Consume optional closing quotes immediately after
                 if end_idx < length and text[end_idx] in ['”', '」', '"']:
                     end_idx += 1
                 
                 # Extract sentence
                 sentence = text[current_start:end_idx]
                 if sentence.strip():
                     sentences.append(sentence)
                 
                 current_start = end_idx
                 i = end_idx - 1 # Adjust for loop increment
                 
        i += 1
        
    # Append remaining text
    if current_start < length:
         remainder = text[current_start:]
         if remainder.strip():
             sentences.append(remainder)
             
    return sentences

def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    ENCODER = tiktoken.get_encoding(model_name)
    results = []
    
    for text in docs:
        sentences = split_text_by_sentences(text)
        
        current_chunk = []
        current_len = 0
        
        # Pre-calculate lengths to avoid repeated encoding if desired, 
        # but encoding sentence by sentence is fine for now
        
        for sentence in sentences:
            tokens = ENCODER.encode(sentence)
            length = len(tokens)
            
            if current_len + length > max_token_size:
                # Chunk full, save it
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    results.append({
                        "hash_code": compute_mdhash_id(chunk_text),
                        "text": chunk_text
                    })
                
                # Handle overlap: Keep recent sentences that fit in overlap_token_size
                back_len = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_tokens = ENCODER.encode(s)
                    s_len = len(s_tokens)
                    if back_len + s_len > overlap_token_size:
                        break
                    overlap_sentences.insert(0, s)
                    back_len += s_len
                
                current_chunk = overlap_sentences
                current_len = back_len
                
                # If the single new sentence is extremely long (> max_token_size), 
                # we must handle it. Current logic will just start a new chunk with it.
                # If it's still too big, we might want to hard-split it, but requirement says "complete sentences".
                # We will accept it as an oversized chunk if it is a single sentence.
                
            current_chunk.append(sentence)
            current_len += length
            
        # Last chunk
        if current_chunk:
             chunk_text = "".join(current_chunk)
             results.append({
                "hash_code": compute_mdhash_id(chunk_text),
                "text": chunk_text
             })

    return results

def semantic_chunk_documents(
    docs,
    model_path,
    max_token_size=512,
    min_token_size=256,
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
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load sentence transformer model or tokenizer: {e}")
        raise e
        
    results = []
    
    # Remove local definition of split_text_by_sentences if it exists
    # And use the global one
    
    for doc_text in docs:
        sentences = split_text_by_sentences(doc_text)
        # sentences = [s.strip() for s in sentences if s.strip()] # functions does this
        
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
        
        sentence_lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
        
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
                         if temp_len >= min_token_size and temp_len <= max_token_size:
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
    del tokenizer
    if 'embeddings' in locals():
        del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

if __name__ == "__main__":
    max_token_size = 2048
    min_token_size = 512
    overlap_token_size = 128
    
    # Configuration
    use_semantic = True
    # embedding_model_path = "E:/AI/LLM/Models/Embedding/Youtu-Embedding" 
    # Use raw string or forward slashes to avoid escape issues
    embedding_model_path = r"/Models/Embedding/Youtu-Embedding"

    
    # 示例输入路径，用户可修改此变量
    # original_text_file = "datasets/mix/mix.jsonl" 
    original_text_file = "./input/test.txt" # 示例: 纯文本输入

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
                        min_token_size=min_token_size,
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