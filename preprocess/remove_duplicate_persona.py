from torch._C import Value
from Persona_Selector import *
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer


def remove_similar(df, distance, threshold):
    distance_df = cosine_similarity(df)
    # max_emb = {}
    # print("shape of df is ", df.shape)
    # print("shape of distance is ", distance_df.shape)
    # # print(df.iloc[0,0])
    # for i in range(distance_df.shape[0]):

    #     distance_df[i][i] = 0.0
    #     max_idx = np.where(distance_df[i]==np.amax(distance_df[i]))[0][0]
    #     distance_df[i][i] = 1.0
    #     if tuple(df.iloc[ max_idx]) not in max_emb.keys():
    #         # max_emb[df.iloc[i]] = df.iloc[max_idx]
    #         max_emb[tuple(df.iloc[max_idx])] = []
    #     max_emb[tuple(df.iloc[max_idx])].append(df.iloc[i])

    # exit(0)
    # print()
    # print("size of max_emb is ", np.shape(max_emb))
    similar_indices = [(x, y) for (x, y) in np.argwhere(distance_df > threshold) if x != y]
    # print(np.shape(similar_indices))
    similar_indices = list(set([item for tpl in similar_indices for item in tpl]))
    # print(np.shape(similar_indices))
    # exit(0)
    new_df = df[~df.index.isin(similar_indices)]
    print(new_df.shape)
    similarity = cosine_similarity(df, new_df)
    similarity_idx = []
    for i in range(len(df)):
        similarity_idx.append(np.where(similarity[i] == np.amax(similarity[i]))[0][0])
    # print(similarity_idx)
    # print(np.shape(similarity_idx))
    # exit(0)
    return new_df, similarity_idx


# model = SentenceTransformer('bert-base-nli-mean-tokens')
def remove_duplicate_persona(persona_pool, threshold=0.7):
    # new_pool is the pool of clean pool by threshold
    # similarity_idx is in shape of (6732, ), type int
    # The ith item implies the corresponding persona idx in the new pool
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("msmarco-distilbert-base-v3")
    # query_embedding = model.encode('How big is London')
    # passage_embedding = model.encode('London has 9,787,426 inhabitants at the 2011 census')

    # print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
    # persona_pool = prepare_persona_selector()

    embedding_table = {}
    for i in range(len(persona_pool)):
        embedding_table[tuple(model.encode(persona_pool[i]))] = persona_pool[i]

    df = pd.DataFrame(embedding_table.keys())
    ori_num = df.shape[0]
    df, similarity_idx = remove_similar(df, util.pytorch_cos_sim, 0.7)
    print(f"Under threshold = {threshold}, we remove {ori_num - df.shape[0]} similar persona")
    print(f"There are {df.shape[0]} persona remained in the pool")
    df = df.to_numpy()
    new_pool = []

    for embedding in df:
        new_pool.append(embedding_table[tuple(embedding)])
    return new_pool, similarity_idx
    # exit(0)
    # np.save("clean_persona.npy", new_pool)


# persona_pool = model.decode(df.to_numpy())
# print(persona_pool)
# print(len(persona_pool))
# print(len(embeddings))
if __name__ == "__main__":
    remove_duplicate_persona()
