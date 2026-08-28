def mixed_chunking(source_text, min_length=25, max_length=350):
   
    paragraphs = source_text.split("\n\n")
    
    final_chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue 
            
        combined = (current_chunk + " " + para).strip()
        combined_words = combined.split()
        
        if len(combined_words) > max_length:
            for i in range(0, len(combined_words), max_length):
                slice_text = " ".join(combined_words[i : i + max_length])
                final_chunks.append(slice_text)
            current_chunk = "" 
            
        elif len(combined_words) < min_length:
            current_chunk = combined
            
        else:
            final_chunks.append(combined)
            current_chunk = ""
            
    if current_chunk:
        final_chunks.append(current_chunk)
        
    return final_chunks


def build_chunk_objs(book_text_obj, chunks):
    """
    Args:
        book_text_obj (dict): A dictionary containing article metadata,
                            including 'title', 'pubDate', 'link', 'guid',
                            and 'description'.
        chunks (list): A list of text chunks derived from the article content.
    """

    chunk_objs = []

    for i, c in enumerate(chunks):
        chunk_obj = {
            'title' : book_text_obj["title"],
            'pubDate' : book_text_obj['pubDate'],
            'link' : book_text_obj['link'],
            'guid' : book_text_obj['guid'],
            'description' : book_text_obj['description'],
            'chunk' : c,
            'chunk_index' : i
        }
        chunk_objs.append(chunk_obj)
    return chunk_objs


def chunk_data(data):
    chunk_objs = [] 

    for obj in data:
        article_chunks = mixed_chunking(obj["article_content"])
        article_chunk_objs = build_chunk_objs(obj, article_chunks)
        chunk_objs.extend(article_chunk_objs)
    return chunk_objs