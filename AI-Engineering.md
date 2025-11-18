# RAG

## Retrieval Algorithms

1. term-based retrieval

    Using keywords. 

    Elasticsearch / BM25

2. embedding-based retrieval 

    Embed both query and external knowledge. It has:

    a. Embedding model

    b. Retriever: fetch relevant embeddings

    For larger scale search: ANN: FAISS, ScaNN, Hnswlib
    
    Rank of retrieved documents: NDCG (normalized discounted cumulative gain), MAP (Mean Average Precision), and MRR (Mean Reciprocal Rank)

    Evaluation:

        1. Retrieval quality

        2. Final RAG outputs

        3. Embeddings

3. hybrid retrieval

## Retrieval Optimization

1. Chunking

    Simple way: equal length

    Set overlapps

2. Reranking

    Can be based on time

3. Query Rewriting

4. Contextual retreval

    Augment chunk with relevant context - metadata

    Especially in chunks, key info could be missing. Can use the original document's title / summary