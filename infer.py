import argparse
import os
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def build_gallery_embeddings(emb_model, gallery_root):
    gallery_vectors = []
    gallery_paths = []

    for root, _, files in os.walk(gallery_root):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                fp = os.path.join(root, f)
                x = load_and_preprocess(fp)
                emb = emb_model(x)[0].numpy()
                gallery_vectors.append(emb)
                gallery_paths.append(fp)

    return np.vstack(gallery_vectors), gallery_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    # load embedding model via TFSMLayer
    emb = TFSMLayer(args.model, call_endpoint="serve")

    # build gallery
    gallery_vecs, gallery_paths = build_gallery_embeddings(emb, args.gallery)

    # load query
    q = load_and_preprocess(args.query)
    q_emb = emb(q)[0].numpy().reshape(1, -1)

    # similarity
    sims = cosine_similarity(q_emb, gallery_vecs)[0]
    topk_idx = sims.argsort()[::-1][:args.topk]

    print("\nTop matches:")
    for idx in topk_idx:
        print(f"{gallery_paths[idx]} (score={sims[idx]:.4f})")

if __name__ == "__main__":
    main()
