
# Reads the PDF and uploads its chunks to Pinecone	
# Manually, when you ingest or re-ingest the PDF


# Import argparse to accept command-line arguments, such as the PDF file path.
import argparse

# Import hashlib to generate a unique, repeatable ID for each text chunk.
import hashlib

# Import Path to check whether the PDF file exists and handle file paths.
from pathlib import Path

# Import PyMuPDF to open PDF files and extract text from each page.
import pymupdf

# Import ensure_index to connect to or create the Pinecone index.
# Import embed to convert text chunks into local embedding vectors.
# The embed() function uses Sentence Transformers, not OpenAI.
from app.knowledge import ensure_index, embed


# Split extracted text into smaller overlapping chunks for retrieval.
def chunks(text: str, size: int = 1100, overlap: int = 180):
    """Split extracted text into smaller overlapping chunks."""

    # Validate the chunk configuration.
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    # Normalize whitespace by replacing newlines and repeated spaces
    # with single spaces.
    text = " ".join(text.split())

    # Calculate how many characters to move forward for each chunk.
    # Overlap preserves some context between consecutive chunks.
    step = size - overlap

    # Generate chunks of at most 'size' characters.
    # Move forward by 'step' characters to create overlapping chunks.
    # Keep only chunks longer than 80 characters.
    return [
        text[i:i + size]
        for i in range(0, len(text), step)
        if len(text[i:i + size]) > 80
    ]


# Upload a batch of document chunks to Pinecone.
def upload_batch(index, batch):
    """Generate local embeddings and upload chunks to Pinecone."""

    # Do nothing if the batch is empty.
    if not batch:
        return

    # Extract the text content from each batch item.
    # The embed() function uses a local Sentence Transformers model.
    # It does not call the OpenAI embeddings API.
    texts = [item[1] for item in batch]

    # Convert the document text into numerical embedding vectors.
    # For all-MiniLM-L6-v2, each vector contains 384 dimensions.
    vectors = embed(texts)

    # Ensure every document chunk has a corresponding embedding.
    if len(vectors) != len(batch):
        raise ValueError(
            "The number of embeddings does not match the batch size."
        )

    # Combine each chunk's ID, embedding vector, and metadata.
    # Pinecone upsert inserts new records or updates records
    # that already exist with the same IDs.
    index.upsert(
        vectors=[
            {
                "id": item[0],
                "values": vector,
                "metadata": {
                    **item[2],
                    "source": item[2]["file"],
                },
            }
            for item, vector in zip(batch, vectors)
        ]
    )


# Main function that runs the complete PDF ingestion pipeline.
def main():

    # Create a command-line argument parser.
    parser = argparse.ArgumentParser(
        description="Ingest a medical PDF into the Pinecone knowledge base."
    )

    # Accept an optional PDF path.
    # If no path is supplied, use data/medical_book.pdf.
    parser.add_argument(
        "--pdf",
        default="data/medical_book.pdf",
        help="Path to the medical PDF file",
    )

    # Read the command-line arguments.
    args = parser.parse_args()

    # Convert the supplied PDF path into a Path object.
    pdf = Path(args.pdf)

    # Stop execution if the specified PDF file does not exist.
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")

    # Connect to the Pinecone index or create it if necessary.
    # The configured index must have dimension 384 when using
    # sentence-transformers/all-MiniLM-L6-v2.
    index = ensure_index()

    # Store chunks temporarily before embedding and uploading them.
    batch = []

    # Keep track of the total number of chunks processed.
    total_chunks = 0

    # Open the PDF file and automatically close it when processing finishes.
    with pymupdf.open(pdf) as doc:

        # Get the total number of PDF pages.
        total_pages = len(doc)

        # Iterate through every PDF page, numbering pages starting at 1.
        for page_number, page in enumerate(doc, start=1):

            # Extract text from the current page.
            page_text = page.get_text()

            # Split the extracted page text into smaller chunks.
            # Number chunks on each page starting at 1.
            for part, content in enumerate(
                chunks(page_text), start=1
            ):

                # Create a deterministic SHA-256 ID using the file name,
                # page number, chunk number, and chunk content.
                # Identical inputs generate the same ID.
                key = hashlib.sha256(
                    (
                        f"{pdf.name}:{page_number}:"
                        f"{part}:{content}"
                    ).encode("utf-8")
                ).hexdigest()

                # Store the chunk ID, text, and metadata in the batch.
                # Metadata lets retrieval results identify the source page
                # and original PDF file.
                metadata = {
                    "page": page_number,
                    "text": content,
                    "file": pdf.name,
                }

                # Append the chunk information to the batch.
                # Each item contains the ID, original text, and metadata.
                batch.append((key, content, metadata))

                # Increment the total number of chunks processed.
                total_chunks += 1

                # Upload chunks in batches of 50 to avoid processing
                # the entire PDF in memory at once.
                if len(batch) >= 50:

                    # Generate local embeddings and upload this batch
                    # to Pinecone.
                    upload_batch(index, batch)

                    # Clear the batch after uploading to prepare for more
                    # chunks.
                    batch.clear()

            # Print progress after every 50 PDF pages.
            # flush=True displays the message immediately in the terminal.
            if page_number % 50 == 0:
                print(
                    f"Processed pages: {page_number}/{total_pages}",
                    flush=True,
                )

        # Upload any remaining chunks that did not fill a complete batch.
        upload_batch(index, batch)

    # Confirm that the PDF ingestion process has finished.
    print(
        f"Ingestion complete. Processed {total_pages} pages "
        f"and uploaded {total_chunks} chunks."
    )


# Run main() only when this file is executed directly.
# Importing this file from another Python module will not automatically run it.
if __name__ == "__main__":
    main()



