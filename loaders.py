from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def BS4HTMLLoader(filepath: str):
    """
    Optimized loader for Doxygen-generated HTML:
    - extracts only doc-content
    - chunks by h1/h2/h3 sections
    - headings included in embeddings
    - handles code fragments and tables
    """

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "lxml")

        # ---- isolate real content ----
        doc = soup.find("div", id="doc-content")
        if not doc:
            return []

        # Remove noise inside doc-content
        for tag in doc(["script", "style", "nav"]):
            tag.decompose()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=120
        )

        documents = []

        current_headings = []
        current_text = []

        def flush():
            if not current_text:
                return

            section_text = (
                "SECTION:\n"
                + "\n".join(current_headings)
                + "\n\n"
                + "\n".join(current_text)
            )

            for chunk in splitter.split_text(section_text):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filepath,
                            "section": current_headings[-1] if current_headings else "root"
                        }
                    )
                )

        # ---- walk content in order ----
        for el in doc.descendants:
            if not hasattr(el, "name"):
                continue

            # ---- headings ----
            if el.name in ["h1", "h2", "h3"]:
                flush()
                title = el.get_text(" ", strip=True)
                level = el.name.upper()
                current_headings = [f"{level}: {title}"]
                current_text = []

            # ---- code ----
            elif el.name in ["pre", "div"] and "fragment" in el.get("class", []):
                code = el.get_text("\n", strip=True)
                if code:
                    current_text.append(f"CODE:\n{code}")

            # ---- tables ----
            elif el.name == "table":
                table_text = el.get_text(" ", strip=True)
                if table_text:
                    current_text.append(f"TABLE:\n{table_text}")

            # ---- paragraphs & lists ----
            elif el.name in ["p", "li"]:
                text = el.get_text(" ", strip=True)
                if text:
                    current_text.append(text)

        flush()
        return documents

    except Exception as e:
        print(f"[ERROR] Doxygen parse failed {filepath}: {e}")
        return []
