import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="fyugp_navigator")

results = collection.query(
    query_texts=["what is the eligibility for honours with research"],
    n_results=2
)

for doc in results["documents"][0]:
    print(doc)
    print("---")