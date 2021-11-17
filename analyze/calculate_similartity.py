import json
from sklearn.metrics.pairwise import cosine_similarity
file = open("./sentence.json")
data = json.load(file)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
similarity = {}
for key in data.keys():
    if key not in similarity.keys():
        similarity[key] = {}
    for persona in data[key]['0'].keys():
        output_0 = model.encode(data[key]['0'][persona])
        output_1 = model.encode(data[key]['1'][persona])
        if persona not in similarity.keys():
            similarity[key][persona] = {0:[], 1: []}
        similarity[key][persona][0].append(cosine_similarity([output_0[0]],output_0[1:]).tolist()[0][0])
        similarity[key][persona][1].append(cosine_similarity([output_1[0]],output_1[1:]).tolist()[0][0])
        # print("Shape : ", output_0.shape)
        # print(cosine_similarity([output_0[0]],output_0[1:])[0])
        # print(cosine_similarity([output_1[0]],output_1[:]))
# print(similarity)
simi_0 = []
simi_1 = []
for key in data.keys():
    # if key not in similarity.keys():
        # similarity[key] = {}
    for persona in data[key]['0'].keys():
        simi_0.extend(similarity[key][persona][0])
        simi_1.extend(similarity[key][persona][1])
import numpy as np
print("Mean of cosine similarity of s2 ", np.mean(simi_0))
print("Std of cosine similarity of s2 ",np.std(simi_0))
print("Mean of cosine similarity of s4 ",np.mean(simi_1))
print("Std of cosine similarity of s4 ",np.std(simi_1))
# with open("similarity.json", "w") as f:
    # json.dump(similarity, f)
    