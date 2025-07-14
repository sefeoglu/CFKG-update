# TACRED Few-Shot DPO Sampler using K-Means Clustering with t-SNE Visualization

import json
import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Templates for generating DPO-style explanations
CORRECT_TEMPLATES = [
    "The relation is {label} because it reflects the relationship in the sentence.",
    "This is clearly a case of {label} due to how the entities are described.",
    "{label} is appropriate here based on the context provided."
]

INCORRECT_TEMPLATES = [
    "The relation is {label}, which is incorrect based on the sentence.",
    "This sentence does not support the label {label}.",
    "{label} does not match the actual relation between the entities."
]

TACRED_LABELS = [
    "per:place_of_birth", "org:top_members/employees", "per:spouse",
    "org:city_of_headquarters", "per:children", "no_relation"
]

FIRE_LABERLS = [
]

def get_sentence(tokens):
    return " ".join(tokens)

def highlight_entities(sentence, subj, obj):
    return sentence.replace(subj, f"<SUBJ>{subj}</SUBJ>").replace(obj, f"<OBJ>{obj}</OBJ>")

def generate_prompt(tokens, subj, obj):
    sentence = get_sentence(tokens)
    highlighted = highlight_entities(sentence, subj, obj)
    return f"Identify the relation between <SUBJ>subject</SUBJ> and <OBJ>object</OBJ> in: \"{highlighted}\""

def generate_chosen(label):
    return random.choice(CORRECT_TEMPLATES).format(label=label)

def generate_rejected(correct_label):
    incorrect = random.choice([l for l in TACRED_LABELS if l != correct_label])
    return random.choice(INCORRECT_TEMPLATES).format(label=incorrect)

def kmeans_sample_per_relation(tacred_data, n_per_class=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    by_relation = defaultdict(list)
    all_embeddings = []
    all_labels = []

    for ex in tacred_data:
        if ex["relation"] != "no_relation":
            by_relation[ex["relation"]].append(ex)

    selected = []

    for relation, examples in by_relation.items():
        if len(examples) <= n_per_class:
            selected.extend(examples)
            continue

        texts = [get_sentence(e["tokens"]) for e in examples]
        embeddings = model.encode(texts)
        all_embeddings.extend(embeddings)
        all_labels.extend([relation] * len(embeddings))

        kmeans = KMeans(n_clusters=n_per_class, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        for cluster_id in range(n_per_class):
            cluster_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
            if cluster_indices:
                center = kmeans.cluster_centers_[cluster_id]
                closest_idx = min(cluster_indices, key=lambda i: (embeddings[i] - center).dot(embeddings[i] - center))
                selected.append(examples[closest_idx])

    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(np.array(all_embeddings))
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(all_labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(all_labels) if l == label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, alpha=0.6, color=colors(i))
    plt.legend()
    plt.title("t-SNE of Sentence Embeddings by Relation")
    plt.savefig("tsne_clusters.png")
    plt.close()

    return selected

def convert_to_dpo_format(examples):
    dpo_data = []
    for ex in examples:
        prompt = generate_prompt(ex["tokens"], ex["subject"], ex["object"])
        chosen = generate_chosen(ex["relation"])
        rejected = generate_rejected(ex["relation"])
        dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return dpo_data

def main():
    with open("tacred.json") as f:
        tacred_data = json.load(f)

    sampled = kmeans_sample_per_relation(tacred_data, n_per_class=5)
    dpo_data = convert_to_dpo_format(sampled)

    with open("feedback_data.json", "w") as f:
        json.dump(dpo_data, f, indent=2)

    print(f"Saved {len(dpo_data)} DPO training examples to feedback_data.json")
    print("Saved t-SNE visualization to tsne_clusters.png")

if __name__ == "__main__":
    main()