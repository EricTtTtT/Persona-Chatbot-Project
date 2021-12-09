import random
import numpy as np
import torch

from sklearn.metrics import classification_report, roc_auc_score
import torch.nn as nn
import os


from transformers import BertConfig
from transformers import AutoTokenizer, BertModel

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
device_1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
# device_1 = torch.device("cuda:0")
mlp_hidden_dim = [64, 32, 8]
epochs = 400
lr = 0.001
batch_size = 100
dropout = 0.8
pooling = "mean"
optimizer = "Adam"
reg = 0.001

train_dir = "./model/engaging_classifier/model/"


class Engagement_cls:
    """This class classifies each query and response pairs as 0(not engaging) or 1 (engaging)"""

    def __init__(
        self,
        train_dir,
        batch_size,
        mlp_hidden_dim,
        num_epochs,
        regularizer=0.01,
        lr=1e-4,
        dropout=0.1,
        optimizer="Adam",
        ftrain_queries_embed=None,
        ftrain_replies_embed=None,
        fvalid_queries_embed=None,
        fvalid_replies_embed=None,
        ftest_queries_embed=None,
        ftest_replies_embed=None,
    ):

        self.train_dir = train_dir
        self.batch_size = batch_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.optim = optimizer
        self.reg = regularizer
        self.query = None
        self.reply = None
        self.query_emb = {}
        self.reply_emb = {}
        model_config = BertConfig(output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", config=model_config)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            self.model.to(device_1)
            self.bert_model.to(device_1)

            # print(device_1)
        self.model.load_state_dict(torch.load(self.train_dir + "best_model_finetuned.pt"))
        info = torch.load(self.train_dir + "best_model_finetuned.info")
        self.model.eval()
        self.bert_model.eval()

    def clean(self):
        self.model.clean()
        self.query_emb.clear()
        self.reply_emb.clear()

    def preprocess_input(self, query, reply):
        self.model.eval()
        self.bert_model.eval()
        # convert query, reply into bert embedding
        # Need to confirm whether need + eos
        MAX_LENGTH = 60
        self.query = query
        self.reply = reply
        Q_vector = []

        masks = []
        for q in self.query:
            q_enc = []
            masks = []

            q_enc.append(self.bert_tokenizer.encode(q))
            masks.append([1 for i in range(len(self.bert_tokenizer.encode(q)))])

            inputs = torch.tensor(q_enc, device=device_1)
            masks = torch.tensor(masks, device=device_1)

            Last_layer_output, _, hidden_state = self.bert_model(inputs, masks)
            last_to_2 = hidden_state[-2]  # the 2 to the last layer output

            q_emb = torch.mean(last_to_2, dim=1)  # Reduce mean
            # print(f"Size of q_emb is {q_emb.size()}")
            self.query_emb[q] = q_emb

        for r in self.reply:
            r_enc = []
            masks = []

            r_enc.append(self.bert_tokenizer.encode(r))
            masks.append([1 for i in range(len(self.bert_tokenizer.encode(r)))])

            inputs = torch.tensor(r_enc, device=device_1)
            masks = torch.tensor(masks, device=device_1)

            Last_layer_output, _, hidden_state = self.bert_model(inputs, masks)
            last_to_2 = hidden_state[-2]  # the 2 to the last layer output

            r_emb = torch.mean(last_to_2, dim=1)  # Reduce mean
            self.reply_emb[r] = r_emb

    def generate_eng_score(self):
        self.model.eval()
        self.bert_model.eval()
        """for all pairs of queries and replies predicts engagement scores
        Params:
            fname_ground_truth: file includes the queries and their ground-truth replies
            foname: file includes the queries, ground truth replies, generated replies (from self.test_replies) and engagement_score of queries and generated replies with following format:
                query===groundtruth_reply===generated_reply===engagement_score of query and generated_reply

        """

        model_output = self.model(self.query, self.reply, self.query_emb, self.reply_emb)
        pred_eng = torch.nn.functional.softmax(model_output, dim=1)

        return pred_eng

    def get_eng_score(self, query, q_embed, reply, r_embed, model):
        self.model.eval()
        self.bert_model.eval()
        """for a pair of query and reply predicts engagement scores
        Params:
            query: input query
            q_embed: embeddings of query
            reply: input reply
            r_embed: embeddings of reply
           
        """
        if not os.path.isfile(self.train_dir + "best_model_finetuned.pt"):
            print("There is not any finetuned model on DD dataset to be used!\nPlease first try to finetune trained model.")
            return

        model_output = self.model(query, reply, q_embed, r_embed)
        # print
        pred_eng = torch.nn.functional.softmax(model_output, dim=1)
        self.query.cpu()
        self.reply.cpu()
        self.query_emb.cpu()
        self.reply_emb.cpu()
        del model_output
        return pred_eng


class BiLSTM(nn.Module):
    """The engagement classification model is a three layer mlp classifier with having tanh as activation functions which takes the embeddings of query and reply as input and pass their average into the mlp classifier"""

    def __init__(self, mlp_hidden_dim=[128], dropout=0.2):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        num_classes = 2
        self.mlp_hidden_0 = nn.Linear(768, mlp_hidden_dim[0], bias=True)
        self.mlp_hidden_1 = nn.Linear(mlp_hidden_dim[0], mlp_hidden_dim[1], bias=True)
        self.mlp_hidden_2 = nn.Linear(mlp_hidden_dim[1], mlp_hidden_dim[2], bias=True)
        self.mlp_out = nn.Linear(mlp_hidden_dim[2], num_classes, bias=True)
        self.X_q = None
        self.X_r = None

    def clean(self):
        self.X_q.cpu().detach()
        self.X_r.cpu().detach()

    def forward(self, queries_input, replies_input, queries_embeds, replies_embeds):

        for q in queries_input:
            # print("Query is ",q)
            # print("Query Embedding is ",queries_embeds[q])
            if q not in queries_embeds.keys():
                print("the query {} embedding has not been found in the embedding file".format(q))
        # X_q = torch.zeros(1,1,60)
        # X_r = torch.zeros(1,1,60)
        self.X_q = torch.tensor([queries_embeds[q].cpu().detach().numpy() for q in queries_input]).squeeze(1).to(device_1)
        # print("Q is ",X_q.size())
        for r in replies_input:
            # print("Query is ",r)
            # print("Reply Embedding is ",replies_embeds[r])
            if r not in replies_embeds.keys():
                print("the reply {} embedding has not been found in the embedding file".format(r))
        self.X_r = torch.tensor([replies_embeds[r].cpu().detach().numpy() for r in replies_input]).squeeze(1).to(device_1)
        # print("R is ",X_r.size())
        if torch.cuda.is_available():
            self.X_q, self.X_r = self.X_q.to(device_1), self.X_r.to(device_1)
        mlp_input = self.X_q.add(self.X_r)
        mlp_input = torch.div(mlp_input, 2)
        # print(mlp_input[0])
        # print(mlp_input[1])
        mlp_h_0 = torch.tanh(self.mlp_hidden_0(mlp_input))
        mlp_h_0 = self.dropout(mlp_h_0)

        mlp_h_1 = torch.tanh(self.mlp_hidden_1(mlp_h_0))
        mlp_h_1 = self.dropout(mlp_h_1)

        mlp_h_2 = torch.tanh(self.mlp_hidden_2(mlp_h_1))
        mlp_h_2 = self.dropout(mlp_h_2)

        mlp_out = self.mlp_out(mlp_h_2)

        # del mlp_h_0. mlp_h_1, mlp_h_2
        return mlp_out


eng_cls = Engagement_cls(train_dir, batch_size, mlp_hidden_dim, epochs, reg, lr, dropout, optimizer)


def analyze_engagement(queries, replies):
    """
    Will return the engagement score
    1 is engaging
    0 is not engaging
    """

    for i in range(len(queries)):
        # print("queries[i] is ", queries[i])
        queries[i] = queries[i].replace("<eos>", "")

    for i in range(len(replies)):
        replies[i] = replies[i].replace("<eos>", "")

    eng_cls.preprocess_input(queries, replies)
    # print("before score")
    scores = eng_cls.generate_eng_score()
    # for q, r, s in zip(queries, replies, score):
    #     print("-------------------------------------------------------------------------")
    #     print(f"Query: {q}")
    #     print(f"Replies: {r}")
    #     print(f"Engaging Score: {s[1]}")
    #     print("-------------------------------------------------------------------------")
    # print("end")
    eng_cls.clean()
    ret = []
    for score in scores:
        ret.append(score[1].item())
    return ret
