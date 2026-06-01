import os

def load_knowledge():
    chunks = []
    folder = "knowledge_base"

    if not os.path.exists(folder):
        print(f"ERROR: The folder '{folder}' does not exist!")
        return chunks

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    file_chunks = content.split("\n\n")
                    file_chunks = [c.strip() for c in file_chunks if c.strip()]
                    chunks.extend(file_chunks)
                    
    print("=== LOADING FILES FROM KNOWLEDGE BASE ===")
    print("TOTAL CHUNKS LOADED:", len(chunks))
    return chunks
