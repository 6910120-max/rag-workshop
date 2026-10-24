"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
from rag.ingest import load_docs

MAX_CHARS = 600


def sections_of(body: str) -> list[tuple[str, list[str]]]:
    """คืน [(ชื่อหัวข้อ, บรรทัดของหัวข้อ)] โดยบรรทัด '## ' เป็นบรรทัดแรกของแต่ละหัวข้อ

    ข้อความก่อนหัวข้อแรกเก็บเป็นหัวข้อที่ชื่อว่างไว้ ส่วนบรรทัด '# ' (ชื่อเอกสาร) ไม่เก็บ
    """
    sections = []
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), [line]))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip() and not line.startswith("# "):
            sections.append(("", [line]))
    return sections


def pack(lines: list[str], max_chars: int) -> list[str]:
    """ต่อบรรทัดเป็นชิ้นไม่เกิน max_chars โดยตัดตามขอบบรรทัด (บรรทัดที่ยาวเกินเองจะตัดตามตัวอักษร)"""
    pieces, cur = [], ""
    for line in lines:
        for i in range(0, max(len(line), 1), max_chars):
            part = line[i:i + max_chars]
            if cur and len(cur) + 1 + len(part) > max_chars:
                pieces.append(cur)
                cur = part
            else:
                cur = f"{cur}\n{part}" if cur else part
    if cur:
        pieces.append(cur)
    return [p.strip() for p in pieces if p.strip()]


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    docs = load_docs() if docs is None else docs
    chunks = []
    for doc in docs:
        meta = {k: v for k, v in doc.items() if k != "body"}
        n = 0
        for heading, lines in sections_of(doc["body"]):
            for text in pack(lines, max_chars):
                n += 1
                chunks.append({
                    **meta,
                    "chunk_id": f"{doc['doc_id']}#{n}",
                    "section": heading or doc["title"],
                    "text": text,
                })
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
