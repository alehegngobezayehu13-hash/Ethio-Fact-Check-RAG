
import re
from collections import Counter

from backend.app.documents import Document, load_documents


def tokenize(text: str) -> list[str]:
    """Split text into words, including words from Ethiopic scripts."""
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


class Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.tokenized = [
            Counter(tokenize(doc.text + " " + doc.title + " " + doc.topic))
            for doc in documents
        ]

    def search(
        self, claim: str, k: int = 3
    ) -> list[tuple[Document, float]]:
        """Return documents ranked by simple word overlap."""
        query_tokens = set(tokenize(claim))
        if not query_tokens:
            return []

        scored = []
        for doc, counts in zip(self.documents, self.tokenized):
            overlap = sum(
                1 for token in query_tokens if counts[token] > 0
            )
            if overlap:
                score = overlap / len(query_tokens)
                scored.append((doc, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:min(k, len(scored))]


if __name__ == "__main__":
    retriever = Retriever(load_documents())
    for claim in [
        "መንግስት አዲስ የትምህርት ፖሊሲ አወጣ",
        "How many months are in the Ethiopian calendar?",
    ]:
        print(f"\nClaim: {claim}")
        for doc, score in retriever.search(claim, k=3):
            print(f"  {score:.3f} [{doc.language}] {doc.text}")
