import numpy as np
import math
import os
import time

import gift as gift
import ascon as ascon

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# ===========================
# XGBoost Hyperparameters
# ===========================

N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0

RANDOM_STATE = 42
N_JOBS = -1

MODEL_DIR = "./saved_models_xgb/"
BLOCK_SIZE = 128

os.makedirs(MODEL_DIR, exist_ok=True)


def train(n, m, num_rounds, bit_size, input_diff):

    print(
        f"Input Parameters --> {CIPHER_NAME} | "
        f"Training Bits: {bit_size} | "
        f"Rounds: {num_rounds}"
    )

    # Select cipher
    if CIPHER_NAME == "GIFT128":
        cipher = gift
    elif CIPHER_NAME == "ASCON320":
        cipher = ascon
    else:
        raise ValueError("Unknown Cipher")

    # Set input difference
    cipher.diff_arr = input_diff

    # Generate training and validation data
    print("Generating training data...")
    X, Y = cipher.make_td_diff(n, 1, num_rounds)

    print("Generating validation data...")
    X_eval, Y_eval = cipher.make_td_diff(m, 1, num_rounds)

    # ----------------------------------------
    # Use the same 40-bit slice as the paper
    # ----------------------------------------
    if CIPHER_NAME == "ASCON320" and bit_size == 40:
        blocks = 8
        k = 7

        start = BLOCK_SIZE * k // blocks
        end = BLOCK_SIZE * (k + 1) // blocks

        X = X[:, start:end]
        X_eval = X_eval[:, start:end]

    print("Training data shape:", X.shape)
    print("Validation data shape:", X_eval.shape)

    # ----------------------------------------
    # XGBOOST Classifier
    # ----------------------------------------
    xgb = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbosity=0
    )

    print("Training XGBOOST...")
    start_time = time.perf_counter()

    xgb.fit(X, Y)
    end_time = time.perf_counter()


    print("Training complete!")
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    pred = xgb.predict(X_eval)

    acc = accuracy_score(Y_eval, pred)

    print(f"Validation Accuracy: {acc:.4f}")

    joblib.dump(xgb, MODEL_DIR + "xgb_ascon.pkl")

    print("Model saved.")


if __name__ == "__main__":

    CIPHER_NAME = "ASCON320"
    BLOCK_SIZE = 320

    train(
        n=2**16,
        m=2**14,
        num_rounds=4,
        bit_size=40,
        input_diff=[
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000001
        ]
    )