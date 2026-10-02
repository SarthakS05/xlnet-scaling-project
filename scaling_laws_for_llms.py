import random

import torch
from torch.optim import Adam
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetLMHeadModel, XLNetTokenizer


def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def build_toy_corpus():
    return [
        "the quick brown fox jumps over the lazy dog",
        "xlnet uses permutation language modeling",
        "autoregressive models capture bidirectional context",
        "deep learning enables powerful nlp systems",
        "transformers rely on self attention mechanisms",
    ]


def random_sentence(corpus):
    return torch.tensor(XLNetTokenizer.from_pretrained("xlnet-base-cased").encode(random.choice(corpus)), dtype=torch.long)


def create_permutation_lm_example(tokenizer):
    ids = torch.tensor(tokenizer.encode(random.choice(build_toy_corpus())), dtype=torch.long)
    length = len(ids)
    perm = torch.randperm(length)
    target = ids.clone()
    input_ids = ids.clone()
    input_ids[:] = tokenizer.pad_token_id

    for i in range(length):
        pos = perm[i]
        input_ids[pos] = ids[pos]

    return input_ids.unsqueeze(0), target.unsqueeze(0)


def pretrain_step(model, tokenizer, optimizer, device):
    input_ids, target = create_permutation_lm_example(tokenizer)
    input_ids = input_ids.to(device)
    target = target.to(device)

    outputs = model(input_ids=input_ids, labels=target)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def encode_classification(tokenizer, text, label, device):
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), torch.tensor([label]).to(device)


def main():
    device = configure_device()
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    config = XLNetConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_head=4,
        n_layer=4,
        d_inner=512,
    )

    model = XLNetLMHeadModel(config).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=3e-4)
    print("Starting XLNet-style toy pretraining...")
    for step in range(200):
        loss = pretrain_step(model, tokenizer, optimizer, device)
        if step % 20 == 0:
            print(f"Step {step} — Loss: {loss:.4f}")

    class_model = XLNetForSequenceClassification(config, num_labels=2).to(device)
    class_model.load_state_dict(model.state_dict(), strict=False)
    class_model.train()

    optimizer_ft = Adam(class_model.parameters(), lr=2e-5)
    classification_data = [
        ("xlnet is powerful", 1),
        ("transformers are great", 1),
        ("the dog slept", 0),
        ("the cat meowed", 0),
    ]

    print("\nStarting XLNet-style fine-tuning...")
    for epoch in range(3):
        total_loss = 0.0
        for text, label in classification_data:
            input_ids, attn_mask, labels = encode_classification(tokenizer, text, label, device)
            outputs = class_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} — Loss: {total_loss:.4f}")

    class_model.eval()
    test_text = "xlnet understands context"
    input_ids, attn_mask, _ = encode_classification(tokenizer, test_text, 0, device)
    pred = class_model(input_ids=input_ids, attention_mask=attn_mask).logits.argmax().item()
    print("\nPrediction for:", test_text)
    print("Label:", pred)


if __name__ == "__main__":
    main()
