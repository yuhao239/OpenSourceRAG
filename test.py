# verify_ingestion.py
# A standalone script to connect to the persisted ChromaDB and verify
# the contents of our ingested data.

import chromadb
from config import Config

def verify_vector_database():
    """
    Connects to the ChromaDB and prints statistics to verify the ingestion.
    """
    print("--- Starting Verification Process ---")
    
    try:
        # 1. Load the application configuration
        config = Config()
        
        # 2. Connect to the persistent ChromaDB client
        print(f"Connecting to ChromaDB at: {config.CHROMA_PERSIST_DIR}")
        db = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        
        # 3. Get the specific collection
        print(f"Attempting to retrieve collection: '{config.CHROMA_COLLECTION_NAME}'")
        collection = db.get_collection(config.CHROMA_COLLECTION_NAME)
        
        # 4. Perform verification checks
        num_items = collection.count()
        print(f"\n--- Verification Results ---")
        if num_items > 0:
            print(f"✅ Success: Collection '{config.CHROMA_COLLECTION_NAME}' found and contains {num_items} embedded chunks.")
            
            # Fetch and display a sample item
            sample_item = collection.peek(limit=1)
            
            print("\n--- Sample Item ---")
            if sample_item.get('ids'):
                print(f"ID: {sample_item['ids'][0]}")
            if sample_item.get('documents'):
                # Display the first 200 characters of the text content
                document_text = sample_item['documents'][0]
                print(f"Document Text (preview): '{document_text[:200]}...'")
            if sample_item.get('metadatas'):
                print(f"Metadata: {sample_item['metadatas'][0]}")
            print("-------------------\n")
            print("✅ Verification PASSED.")
            
        else:
            print(f"❌ Failure: Collection '{config.CHROMA_COLLECTION_NAME}' was found but is empty.")
            print("Please run the ingestion workflow in main.py first.")

    except ValueError as e:
        print(f"\n❌ ERROR: Could not find the collection '{config.CHROMA_COLLECTION_NAME}'.")
        print("This likely means the ingestion workflow has not been run successfully yet.")
        print("Please run `python main.py` to create and populate the database.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

    print("--- Verification Process Finished ---")


if __name__ == "__main__":
    verify_vector_database()
