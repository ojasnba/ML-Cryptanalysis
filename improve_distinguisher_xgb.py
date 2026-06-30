import numpy as np
import joblib
import statistics
import math
import os
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime

import gift as gift
import ascon as ascon

if not os.path.exists("./graphs"):
    os.makedirs("./graphs/")

# Load model globally to save I/O overhead and unlock multi-threading
MODEL_PATH = "./saved_models_xgb/xgb_ascon.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    model.n_jobs = -1  # CRITICAL: Utilizes all CPU cores for parallel predictions
else:
    model = None
        
def prediction(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,accuracy1,accuracy0,bit_size,loop):
    print("Input Parameters--> Model Trained on "+str(bit_size)+" output bits of "+str(CIPHER_NAME)+" | Input Difference for Data:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Input Difference for ML model:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper()+ " | Prediction Data: 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | TP_Real Accuracy: " + str(accuracy1) + " | TP_Random Accuracy: " + str(accuracy0) + " | Number of Experiments: " + str(2*loop)+"\n")
    if (CIPHER_NAME=="GIFT128"):
        cipher = gift
        acc_precision = 2; 
    elif (CIPHER_NAME=="ASCON320"):
        cipher = ascon
        acc_precision = 16;
        
    if (bit_size < BLOCK_SIZE):
        if (CIPHER_NAME=="GIFT128"):
            model_index = 0;
            bit_range = list(range(0,BLOCK_SIZE,8)) # Precomputed list
        elif (CIPHER_NAME=="ASCON320"):
            model_index = 7;
            bit_range = list(range(7*BLOCK_SIZE//8,BLOCK_SIZE)) # Precomputed list
    else:
        model_index = 8
            
    cipher.diff_arr = input_diff_ML
    from sklearn.metrics import accuracy_score

    print("Calculating Accuracies for Real and Random Case: ")

    X, Y = cipher.make_td_diff(n, 1, ml_rounds, data=2)
    X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
    
    pred = model.predict(X_test)
    accuracy2 = accuracy_score(Y, pred)

    print("Accuracy =", accuracy2)
        
    print("All Real Case: ")
    TP_predictions_count = []
    for j in range(0,10):
        X, Y = cipher.make_td_diff( n , 1 , ml_rounds,data=1)
        X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
        X_test = X_test.astype(np.float32, copy=False)
        P = model.predict_proba(X_test)[:, 1]
        TP_predictions_count.append(np.sum(P > 0.5) / n) # Fast vectorization
        
    accuracy1 = min(TP_predictions_count)
    print("TP Accuracy: " + str(accuracy1))
    
    print("All Random Case: ")
    TP_predictions_count = []
    for j in range(10):
        X, Y = cipher.make_td_diff(n, 1, ml_rounds, data=0)
        X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
        P = model.predict_proba(X_test)[:, 1]
        TP_predictions_count.append(np.sum(P > 0.5) / n)

    accuracy0 = max(TP_predictions_count)
    print("TP Accuracy:", accuracy0)

    if accuracy1 == 0 and accuracy0 == 0:
        return accuracy1, accuracy0

    # -------------------------------------------------
    # Distinguisher Construction Phase
    # -------------------------------------------------
    cipher.diff_arr = input_diff
    TP_arr = []

    print("Making Prediction for All Real Data:")
    for l in trange(0, loop):
        X, Y = cipher.make_td_diff(n, 1, num_rounds, data=1)
        X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
        X_test = X_test.astype(np.float32, copy=False)
        P = model.predict_proba(X_test)[:, 1]
        TP_arr.append(np.sum(P > 0.5))

    print("Making Prediction for All Random Data: ")
    for l in trange(0, loop):
        X, Y = cipher.make_td_diff(n, 1, num_rounds, data=0)
        X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
        X_test = X_test.astype(np.float32, copy=False)
        P = model.predict_proba(X_test)[:, 1]
        TP_arr.append(np.sum(P > 0.5))

    print("Prediction Results (All Real Data): ")
    print(TP_arr[0:loop])
    print("Prediction Results (All Random Data): ")
    print(TP_arr[loop:])
    avg_real = statistics.mean(TP_arr[0:loop])
    avg_random = statistics.mean(TP_arr[loop:])
    diff = avg_real - avg_random

    min_real = min(TP_arr[0:loop])
    max_random = max(TP_arr[loop:])

    cutoff_graph = (min_real + max_random) // 2
    cutoff_cal = math.ceil(max(2, n * accuracy0 + (n // class_data) * (accuracy1 - accuracy0) * 0.5))
    
    print("Average TP in All Real Data:", avg_real)
    print("Average TP in All Random Data:", avg_random)
    print("Difference in Average TP Data (Real-Random):", diff)
    print("Min TP in All Real Data:", min_real)
    print("Max TP in All Random Data:", max_random)
    print("Difference in Min_Real-Max_Random: " + str(min_real - max_random))

    print("Graph cutoff:", cutoff_graph)
    print("Calculated cutoff:", cutoff_cal)
    print("Difference in Accuracy: " + str(accuracy1 - accuracy0) + " | Data Difference Expected (based on accuracy): " + str((n // class_data) * (accuracy1 - accuracy0)) + " | Cutoff Using Graph (Algo 3): " + str(cutoff_graph) + " | Cutoff Calculated (Algo 4): " + str(cutoff_cal) + " | Data Used: 2^" + str(int(math.log(n, 2))))

    print("\nResults for Cutoff Using Graph (Algo 3):  " + str(cutoff_graph))
    graph_accuracy = cal_accuracy(TP_arr, cutoff_graph, loop)

    print("\nResults for Cutoff Calculated (Algo 4):  " + str(cutoff_cal))
    calc_accuracy = cal_accuracy(TP_arr, cutoff_cal, loop)

    xdata = list(range(1, loop + 1))
    ydata_1 = TP_arr[0:loop]
    ydata_0 = TP_arr[loop:2 * loop]

    plt.figure()
    plt.title("Rounds: " + str(num_rounds) + " | Data Required: 2^" + str(int(math.log(n, 2))))
    plt.xlabel("Experiment No.")
    plt.ylabel("No. of Prediction > 0.5")
    plt.plot(xdata, ydata_1, 'o', label="TP", linestyle=":", color="green")
    plt.plot(xdata, ydata_0, 'd', label="TN", linestyle=":", color="red")
    os.makedirs("xgb_results//graphs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xgb_results/graphs/{CIPHER_NAME}_{bit_size}bits_{num_rounds}rounds_2^{int(math.log(n,2))}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Graph saved to: {filename}")
    plt.close()
    os.makedirs("xgb_results", exist_ok=True)

    with open("xgb_results/results.txt", "a") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Cipher: {CIPHER_NAME}\n")
        f.write(f"Rounds: {num_rounds}\n")
        f.write(f"Bit Size: {bit_size}\n")
        f.write(f"Data Complexity: 2^{int(math.log(n,2))}\n\n")
        f.write(f"Model Accuracy: {accuracy2:.6f}\n")
        f.write(f"TP Accuracy (Real): {accuracy1:.6f}\n")
        f.write(f"TP Accuracy (Random): {accuracy0:.6f}\n\n")
        f.write(f"Graph Accuracy: {graph_accuracy:.2f}%\n")
        f.write(f"Calculated Accuracy: {calc_accuracy:.2f}%\n")
        f.write(f"Average TP (Real): {avg_real}\n")
        f.write(f"Average TP (Random): {avg_random}\n")
        f.write(f"Difference: {diff}\n")
        f.write(f"Min TP (Real): {min_real}\n")
        f.write(f"Max TP (Random): {max_random}\n\n")
        f.write(f"Graph Cutoff: {cutoff_graph}\n")
        f.write(f"Calculated Cutoff: {cutoff_cal}\n")
        f.write("=" * 60 + "\n\n")
        
    return calc_accuracy

def cal_accuracy(TP_arr, cutoff, loop):
    TP_arr = np.array(TP_arr)
    # Vectorized computation of accuracies instead of slow python elements loop
    real_correct = np.sum(TP_arr[:loop] > cutoff)
    random_correct = np.sum(TP_arr[loop:] <= cutoff)

    accuracy = (real_correct + random_correct) * 100 / (2 * loop)
    print(f"TP_Real Count: {real_correct} | TP_Random Count: {random_correct} | Accuracy: {accuracy}%")
    return accuracy

def distinguisher(n,num_rounds,ml_rounds,class_data,acc,input_diff,input_diff_ML,bit_size,loop,validate,beta,C_T):
    if validate:
        return validation(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,bit_size,C_T,loop)
    return prediction(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,0,0,bit_size,loop)

def validation(n,input_diff,input_diff_ML,num_rounds,class_data,ml_rounds,acc,bit_size,C_T,loop):
    print("Input Parameters--> Model Trained on "+str(bit_size)+" output bits of "+str(CIPHER_NAME)+" | Input Difference for Data:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff])).upper()+ " | Input Difference for ML model:  " + str("".join([hex(i)[2:].zfill(2) for i in input_diff_ML])).upper()+ " | Data Required (beta): 2^" + str(int(math.log(n,2)))+" | Classical Data: 2^" + str(int(math.log(class_data,2))) + " | Total Number of Rounds: " + str(num_rounds) +  " | ML Number of Rounds: " + str(ml_rounds) + " | Model Accuracy: " + str(acc) + " | Cutoff (C_T): " + str(C_T)+ " | Number of Experiments: " + str(2*loop)+"\n")
    
    if (CIPHER_NAME=="GIFT128"):
        cipher = gift
        acc_precision = 2; 
    elif (CIPHER_NAME=="ASCON320"):
        cipher = ascon
        acc_precision = 16;
        
    if (bit_size < BLOCK_SIZE):
        if (CIPHER_NAME=="GIFT128"):
            model_index = 0;
            bit_range = list(range(0,BLOCK_SIZE,8))
        elif (CIPHER_NAME=="ASCON320"):
            model_index = 7;
            bit_range = list(range(7*BLOCK_SIZE//8,BLOCK_SIZE))
    else:
        model_index = 8
    
    print("XGBoost model ready.\n")
    cipher.diff_arr = input_diff
    TP_arr = []
    
    print("Making Prediction for All Real Data: ")
    for l in trange(0,loop):
        X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=1)
        X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
        P = model.predict_proba(X_test)[:, 1]
        TP_predictions_score = np.sum(P)
        if l == 0:
            print("\nREAL DATA\nMin probability:", np.min(P), "\nMax probability:", np.max(P), "\nMean probability:", np.mean(P), "\nTP count:", TP_predictions_count)
        TP_arr.append(TP_predictions_score)
        
    print("Making Prediction for All Random Data: ")
    for l in trange(0,loop):
        X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=0)
        X_test = X[:, bit_range] if (bit_size < BLOCK_SIZE) else X
        P = model.predict_proba(X_test)[:, 1]
        TP_predictions_count = int(np.sum(P > 0.5))
        if l == 0:
            print("\nRANDOM DATA\nMin probability:", np.min(P), "\nMax probability:", np.max(P), "\nMean probability:", np.mean(P), "\nTP count:", TP_predictions_count)
        TP_arr.append(TP_predictions_count)
        
    print("\nResults for Manual Cutoff (For Validation):  " + str(C_T))
    return cal_accuracy(TP_arr,C_T,loop) 

if __name__ == '__main__':
    CIPHER_NAME = "ASCON320"
    if (CIPHER_NAME=="GIFT128"):
        BLOCK_SIZE = 128
        num_rounds = 8
        ml_rounds = 6
        class_data = 2**10
        bit_size = 128
        if (bit_size==BLOCK_SIZE):
            acc_arr = [0,0,0,0,0,.83,.58]
        elif(bit_size==16):
            acc_arr = [0,0,0,0,0,.72,.56,.51]
        acc = acc_arr[ml_rounds]
        input_diff =  [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x00]
        input_diff_ML = [0x00,0x02,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
        loop = 50
        n_test = 2 ** 16
        validate = True
        beta = 2 ** 18
        C_T = 112189
        
    elif (CIPHER_NAME=="ASCON320"):
        BLOCK_SIZE = 320
        num_rounds = 4
        ml_rounds = 4
        class_data = 2**0
        bit_size = 40
        if (bit_size==BLOCK_SIZE):
            acc_arr = [0,0,0,0,0.5027949810028076]
        elif(bit_size==40):
            acc_arr = [0,0,0,0,0.5022529959678650]
        acc = acc_arr[ml_rounds]
        input_diff = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
        input_diff_ML = [0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000001]
        loop = 50
        n_test = 2 ** 19
        validate = False
        beta = 2 ** 18
        C_T = 125161
        
    distinguisher(n_test,num_rounds,ml_rounds,class_data,acc,input_diff,input_diff_ML,bit_size,loop,validate,beta,C_T)