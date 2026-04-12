import logging

logger = logging.getLogger("chunker")

class SemanticChunker:
    @staticmethod
    def chunk_text(text: str, max_words: int = 500, overlap: int = 50) -> list:
        """
        Splits a massive text into a contiguous Array of overlapping semantic partitions.
        Uses a slider mapping perfectly mathematically natively preventing Token truncation.
        
        Args:
            text: Raw string
            max_words: Maximum approximate words (tokens) per chunk
            overlap: Amount of trailing words to preserve dynamically
            
        Returns:
            list of chunk strings
        """
        if not text:
            return []
            
        words = text.split()
        if len(words) <= max_words:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_str = " ".join(words[start:end])
            chunks.append(chunk_str)
            
            # If we reached exactly the end smoothly, mathematically terminate safely
            if end == len(words):
                break
                
            # Slide backward identically properly preserving the overlapping window logically natively
            start += (max_words - overlap)
            
        return chunks
