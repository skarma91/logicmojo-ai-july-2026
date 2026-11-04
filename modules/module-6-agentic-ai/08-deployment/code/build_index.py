"""Bake the corpus vectors into the image (run by the Dockerfile at build time).

Downloads the FastEmbed MiniLM model once (into FASTEMBED_CACHE_PATH, inside the
image) and writes data/embeddings.npy plus data/ids.json. The running service then
only embeds queries, and never downloads anything at startup.
"""

import retriever

if __name__ == "__main__":
    n = retriever.build_index()
    print(f"baked {n} passage vectors into {retriever.EMB.name} and {retriever.IDS.name}")
