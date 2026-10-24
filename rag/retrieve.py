"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json
import re

import numpy as np
from pythainlp import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunk import chunk_docs
from rag.synonyms import expand_query

# คำช่วยในประโยคคำถามที่ไม่ช่วยแยก chunk (ไม่ใช้รายการ stopword ของ pythainlp เพราะตัดคำอย่าง "วัน" "ผิด" ที่สำคัญในเอกสาร)
QUESTION_WORDS = {"กี่", "ไหม", "หรือไม่", "อะไร", "เท่าไร", "เท่าไหร่", "อย่างไร", "ได้", "ต้อง", "ที่", "ของ", "และ", "ครับ", "ค่ะ"}


def tokenize(text: str) -> list[str]:
    """ตัดคำด้วย pythainlp (newmm) ทิ้งช่องว่าง เครื่องหมาย และคำช่วยของประโยคคำถาม"""
    tokens = word_tokenize(text.lower(), engine="newmm")
    return [t for t in tokens if re.search(r"\w", t) and t not in QUESTION_WORDS]


def _scale(scores: np.ndarray) -> np.ndarray:
    top = scores.max()
    return scores / top if top > 0 else scores


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf", alpha: float = 0.7):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)

        hybrid = alpha * TF-IDF ตัวอักษร + (1 - alpha) * BM25 บนคำที่ตัดด้วย pythainlp
        (แต่ละคะแนนหารด้วยค่าสูงสุดของคำค้นนั้นก่อนรวม) และขยายคำค้นด้วย rag/synonyms.py
        """
        if method not in ("tfidf", "hybrid"):
            raise ValueError(f"ไม่รู้จัก method {method!r} (ใช้ tfidf หรือ hybrid)")
        self.method = method
        self.alpha = alpha
        self.chunks = chunks if chunks is not None else chunk_docs()
        texts = [f"{c['title']} {c['section']} {c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(texts)
        if method == "hybrid":
            self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def _scores(self, query: str) -> np.ndarray:
        if self.method == "tfidf":
            return cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        query = expand_query(query)
        char = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        word = np.asarray(self.bm25.get_scores(tokenize(query)))
        return self.alpha * _scale(char) + (1 - self.alpha) * _scale(word)

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        scores = self._scores(query)
        results = []
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            c = self.chunks[i]
            if not include_superseded and c["status"] != "active":
                continue
            results.append({**c, "score": float(scores[i])})
            if len(results) == k:
                break
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="tfidf", choices=["tfidf", "hybrid"])
    ap.add_argument("--include-superseded", action="store_true", help="ค่าเริ่มต้นตัดฉบับ superseded ออก")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = Retriever(method=a.method).search(a.query, k=a.k, include_superseded=a.include_superseded)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    for r in res:
        print(f"{r['score']:.3f}  {r['chunk_id']}  [{r['status']}]  {r['title']} / {r['section']}")
        print(f"    {r['text'][:120]}")


if __name__ == "__main__":
    main()
