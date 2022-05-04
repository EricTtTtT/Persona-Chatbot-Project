import os
import csv
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
from train import get_data_loaders_persona


from Engaging_classifier import analyze_engagement

from DialogRPT.src.score import get_coherence_score


SPECIAL_TOKENS = ["<bos>", "<|eos|>", "<speaker1>", "<speaker2>", "<pad>"]

from chatbot_turn import generate_response, prepare_chatbot



def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--gpt2_persona_checkpoint", type=str, default="model/gpt2_persona_model/")
    parser.add_argument("--save_dir", type=str, default="model/")
    parser.add_argument("--model_name", type=str, default="new_gen_d13his_tune")
    parser.add_argument("--output_csv", type=str, default="each_persona.csv")


    # hyper parameters
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--dataset_size", type=int, default=1024)

    
    args = parser.parse_args()
    args.model_save_folder = os.path.join(args.root, args.save_dir, args.model_name)

    if os.path.exists(args.output_csv):
        print(f"output csv file {args.output_csv} already exists!!")
        ex = input("overwrite? (y/n)")
        if ex != "n":
            os.remove(args.output_csv)

    # ===== prepare dataset, models and optimizer ==========
    model, interlocutor, tokenizer, arg = prepare_chatbot(
        os.path.join(args.root, args.gpt2_persona_checkpoint), bt=args.batch_size
    )

    # persona_pool = remove_duplicate_persona()
    persona_pool = np.load("./clean_persona.npy")
    print("shape of persona_pool", np.shape(persona_pool))

    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders_persona(arg, tokenizer, True, args.dataset_size)
    del val_loader, train_sampler, valid_sampler

    print(
        """
        ######################################################
        finish preparing  !!!!!!!!!!!!!!!!!
        ######################################################
    """
    )

    # header = ["persona_id", "persona", "engaging_mean", "engaging_std", "coherence_mean", "coherence_std", "sum"]
    header = ["persona_id", "persona", "engaging", "coherence", "sum"]
    with open(args.output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    num_batch = args.dataset_size / args.batch_size
    for i_persona, bot_persona in enumerate(persona_pool):
        print(f"persona {i_persona}", bot_persona)
        engaging_score_sum = 0
        coherence_score_sum = 0
        persona_bot_enc = [tokenizer.encode(bot_persona) for _ in range(args.batch_size)]
        with open(args.output_csv, "a") as f:
            writer = csv.writer(f)
            for inter_persona_ori, history_ori, len_p, len_h in tqdm(train_loader):
                # recover inter_persona and history from padded datum
                inter_persona_enc = []
                for p_ori, lens in zip(inter_persona_ori, len_p):
                    l = sum(lens)
                    inter_persona_enc.append(p_ori[:l].tolist())
                history_enc = []
                for h_ori, lens in zip(history_ori, len_h):
                    tmp = []
                    j = 0
                    for l in lens:
                        if l > 0:
                            tmp.append((h_ori[j : j + l]).tolist())
                            j += l
                    history_enc.append(tmp)

                history = [[tokenizer.decode(s) for s in h] for h in history_enc]
                with torch.no_grad():
                    # chatbots
                    response_enc = generate_response(persona_bot_enc, history_enc, tokenizer, model, arg)
                    history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]

                    # interlocutor
                    response_enc = generate_response(inter_persona_enc, history_enc, tokenizer, interlocutor, arg)
                    history_enc = [h + [r] for h, r in zip(history_enc, response_enc)]

                last_history = [h[-1] if len(h) > 0 else "" for h in history]
                bot_reply = [tokenizer.decode(h[-2]) for h in history_enc]
                inter_reply = [tokenizer.decode(h[-1]) for h in history_enc]

                score = analyze_engagement(bot_reply, inter_reply)
                coherence = get_coherence_score(last_history, bot_reply)
                engaging_score_sum += sum(score) / len(score)
                coherence_score_sum += sum(coherence) / len(coherence)

            engaging_mean = engaging_score_sum / num_batch
            coherence_mean = coherence_score_sum / num_batch
            writer.writerow([i_persona, bot_persona, engaging_mean, coherence_mean, engaging_mean + coherence_mean])


            #     engaging_score.extend(score)
            #     coherence_score.extend(coherence)

            # engaging_mean = sum(engaging_score)/len(engaging_score)
            # engaging_std = 0
            # for s in engaging_score:
            #     engaging_std += (s - engaging_mean)**2
            # engaging_std = np.sqrt(engaging_std/len(engaging_score))
            # coherence_mean = sum(coherence_score)/len(coherence_score)
            # coherence_std = 0
            # for s in coherence_score:
            #     coherence_std += (s - coherence_mean)**2
            # coherence_std = np.sqrt(coherence_std/len(coherence_score))
            # writer.writerow([i_persona, bot_persona, engaging_mean, engaging_std, coherence_mean, coherence_std, engaging_mean + coherence_mean])

if __name__ == "__main__":
    main()
