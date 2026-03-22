"""
Debug script: Check what text is being extracted from key pages
Run this with: python debug_extract.py <path_to_pdf>
"""
import sys
import pdfplumber

def extract_words_text(page):
    words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
    if not words:
        return ""
    lines = []
    current_line = [words[0]]
    for w in words[1:]:
        if abs(w["top"] - current_line[-1]["top"]) < 5:
            current_line.append(w)
        else:
            current_line.sort(key=lambda x: x["x0"])
            line_text = " ".join(w["text"] for w in current_line)
            lines.append(line_text)
            current_line = [w]
    if current_line:
        current_line.sort(key=lambda x: x["x0"])
        line_text = " ".join(w["text"] for w in current_line)
        lines.append(line_text)
    return "\n".join(lines)

if len(sys.argv) < 2:
    print("Usage: python debug_extract.py ~/Downloads/IAR_2025.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]

# Check pages 1 (cover), 11-14 (Key Highlights / Sustainability)
check_pages = [0, 1, 10, 11, 12, 13, 14]

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}\n")

    for p in check_pages:
        if p >= len(pdf.pages):
            continue
        page = pdf.pages[p]
        print(f"\n{'='*80}")
        print(f"PAGE {p+1}")
        print(f"{'='*80}")

        # Method 1
        text1 = page.extract_text()
        print(f"\n--- extract_text() [{len(text1) if text1 else 0} chars] ---")
        if text1:
            print(text1[:500])
        else:
            print("[EMPTY]")

        # Method 2
        text2 = extract_words_text(page)
        print(f"\n--- extract_words() [{len(text2) if text2 else 0} chars] ---")
        if text2:
            print(text2[:500])
        else:
            print("[EMPTY]")

        # Check for key terms
        combined = (text1 or "") + " " + (text2 or "")
        combined_lower = combined.lower()

        key_terms = ["710", "employees", "operating revenue", "701.8", "patami",
                     "250.2", "bursa malaysia berhad", "key highlights"]
        found = [t for t in key_terms if t in combined_lower]
        if found:
            print(f"\n🔍 FOUND KEY TERMS: {found}")
        else:
            print(f"\n❌ No key terms found on this page")
