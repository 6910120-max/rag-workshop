"""พจนานุกรมคำพ้องสำหรับขยายคำค้นของ method="hybrid"

คีย์คือคำ/วลีที่ผู้ใช้มักพิมพ์ ค่าคือคำที่เอกสารบริษัทใช้จริง
รายการทั้งหมดได้มาจากคำถามที่พลาดในชุด dev เท่านั้น (ห้ามเพิ่มจากชุด test)
"""

SYNONYMS: dict[str, list[str]] = {
    # Q24 (dev, paraphrase): ผู้ใช้เขียนทับศัพท์ แต่เอกสาร IT-01 ใช้ "รหัสผ่าน"
    "พาสเวิร์ด": ["รหัสผ่าน"],
    # Q35 (dev, paraphrase): เอกสาร FA-01 ใช้ "รถยนต์ไฟฟ้า" ไม่มีคำว่า "อีวี"
    "อีวี": ["รถยนต์ไฟฟ้า", "จุดชาร์จ"],
}


def expand_query(query: str) -> str:
    """ต่อคำที่เอกสารใช้ท้ายคำค้น เมื่อคำค้นมีคำพ้องอยู่ในพจนานุกรม"""
    q = query.lower()
    extra = [w for key, words in SYNONYMS.items() if key.lower() in q for w in words]
    return f"{query} {' '.join(extra)}" if extra else query
