import re

KNOWN_HEADINGS = [
    "DOCUMENT TYPE:",
    "CORE ALIASES:",
    "DIAGNOSTIC CRITERIA:",
    "REQUIRED FEATURES:",
    "RULE-OUTS / DIFFERENTIALS:",
    "COMMON COMORBIDITIES:",
    "RED FLAGS:",
    "FIRST-LINE TREATMENT:",
    "SECOND-LINE / ESCALATION:",
    "BSV / VOCAL SIGNALS:",
    "PATIENT-LANGUAGE EXAMPLES:",
    "COUNTER-SIGNALS / NEGATIONS:",
    "PROVENANCE:"
]

def split_into_sections(text):
    """
    Split content into (heading, section_text) tuples using known headings.
    Falls back to whole text if no headings found.
    """
    pattern = "|".join(re.escape(h) for h in KNOWN_HEADINGS)
    parts = re.split(f"({pattern})", text, flags=re.IGNORECASE)
    
    sections = []
    current_heading = "BODY"
    current_text = ""
    
    for part in parts:
        if not part:
            continue
        upper_part = part.strip().upper()
        if any(upper_part.startswith(h.upper()) for h in KNOWN_HEADINGS):
            if current_text.strip():
                sections.append((current_heading, current_text.strip()))
            current_heading = part.strip().rstrip(":")
            current_text = ""
        else:
            current_text += part
    
    if current_text.strip():
        sections.append((current_heading, current_text.strip()))
    
    if not sections:
        sections = [("BODY", text.strip())]
    return sections

def chunk_section(section_text, heading, chunk_size=400, overlap=50):
    """
    Chunk a single section, prepending the heading to preserve context.
    """
    if not section_text:
        return []
    words = section_text.split()
    chunks = []
    prefix = f"[{heading}] "
    effective_size = chunk_size - len(prefix.split())
    
    for i in range(0, len(words), effective_size - overlap):
        chunk_words = words[i : i + effective_size]
        chunk = prefix + " ".join(chunk_words)
        chunks.append(chunk)
        if i + effective_size >= len(words):
            break
    return chunks

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Heading-aware chunking. Splits on known clinical headings first,
    then word-chunks within sections while preserving heading context.
    """
    sections = split_into_sections(text)
    all_chunks = []
    for heading, section_text in sections:
        section_chunks = chunk_section(section_text, heading, chunk_size, overlap)
        all_chunks.extend(section_chunks)
    return all_chunks

def process_documents(documents, chunk_size=400, overlap=50):
    """
    Convert a list of documents into a list of chunks with metadata.
    Now heading-aware for better retrieval alignment with clinical sections.
    """
    all_chunks = []
    for doc in documents:
        text = doc.get("content", "")
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        for i, chunk_text_content in enumerate(text_chunks):
            chunk = {
                "chunk_id": f"{doc.get('source', 'UNK')}_{doc.get('condition', 'UNK')}_{i}".replace(" ", "_"),
                "source": doc.get("source", "Unknown"),
                "condition": doc.get("condition", "Unknown"),
                "text": chunk_text_content,
                "title": doc.get("title", "Unknown")
            }
            all_chunks.append(chunk)
    return all_chunks

if __name__ == "__main__":
    from corpus_loader import load_corpus
    docs = load_corpus("data/corpus")
    chunks = process_documents(docs)
    print(f"Generated {len(chunks)} chunks.")
    if chunks:
        print(f"Sample Chunk 0: {chunks[0]['chunk_id']}")
        print(f"Content: {chunks[0]['text'][:100]}...")
