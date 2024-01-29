import os
import json
import numpy as np
import pandas as pd
from typing import List


def load_defender_embeddings():
    docs = []
    for filename in os.listdir("defender_docs"):
        filepath = os.path.join("defender_docs", filename)
        with open(filepath, "r") as file:
            docs.append(json.load(file))
    return pd.DataFrame(docs)


defender_docs = load_defender_embeddings()


def search_defender_doc(user_inquiry_embedding: List[float]):
    # closest embedding by dot product
    dot_products = np.dot(np.stack(defender_docs["embedding"]), user_inquiry_embedding)
    index = np.argmax(dot_products)

    # return the document
    return defender_docs.iloc[index]
