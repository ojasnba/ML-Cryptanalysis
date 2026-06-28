import numpy as np
import joblib
import statistics
import math
from tqdm import trange

import ascon as ascon

CIPHER_NAME = "ASCON320"
BLOCK_SIZE = 320

def cal_accuracy(TP_arr,cutoff,loop):
    TP_Real_count = 0
    TP_Random_count = 0
    for i in range(0,len(TP_arr)):
        if (TP_arr[i] > cutoff):
            if (i< loop):
                TP_Real_count += 1
        else:
            if (i >= loop):
                TP_Random_count += 1
    print("TP_Real Count: " + str(TP_Real_count) + " | TP_Random Count: " + str(TP_Random_count) + " | Accuracy: " + str(TP_Real_count+TP_Random_count)+str("%"))
    return (TP_Real_count+TP_Random_count)*100/(2*loop)

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
            bit_range = range(0,BLOCK_SIZE,8)
        elif (CIPHER_NAME=="ASCON320"):
            model_index = 7;
            bit_range = range(7*BLOCK_SIZE//8,BLOCK_SIZE)
    else:
        model_index = 8
    
    
    cipher.diff_arr = input_diff_ML
    model = joblib.load("./saved_models_rf/rf_ascon.pkl")
    print("Random Forest model loaded.\n")
    cipher.diff_arr = input_diff
    TP_arr = []
    print("Making Prediction for All Real Data: ")
    for l in trange(0,loop):
        X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=1)
        if (bit_size < BLOCK_SIZE):
            X_test = X[:, [i for i in bit_range]]
        else:
            X_test = X
        P = model.predict_proba(X_test)[:, 1]
        TP_predictions_count = int(np.sum(P > 0.5))
        TP_arr.append(TP_predictions_count)
    print("Making Prediction for All Random Data: ")
    for l in trange(0,loop):
        X, Y = cipher.make_td_diff( n , 1 , num_rounds,data=0)
        if (bit_size < BLOCK_SIZE):
            X_test = X[:, [i for i in bit_range]]
        else:
            X_test = X
        P = model.predict_proba(X_test)[:, 1]
        TP_predictions_count = int(np.sum(P > 0.5))
        TP_arr.append(TP_predictions_count)
    print("\nResults for Manual Cutoff (For Validation):  " + str(C_T))
    return cal_accuracy(TP_arr,C_T,loop)

if __name__ == "__main__":

    CIPHER_NAME = "ASCON320"
    BLOCK_SIZE = 320

    input_diff = [
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000001
    ]

    validation(
        n=2**18,
        input_diff=input_diff,
        input_diff_ML=input_diff,
        num_rounds=4,
        class_data=2**0,
        ml_rounds=4,
        acc=0.5022529959678650,
        bit_size=40,
        C_T=125161,
        loop=50
    )