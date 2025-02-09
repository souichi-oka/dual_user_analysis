import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import glob
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import scipy as sp
import csv


def parse_args() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("IN_DIR", help="Input dir", type=str)

    args = parser.parse_args()
    return (args.IN_DIR)

def add_dual_col(df) -> pd.DataFrame:
    df["SMKDUAL"] = 0

    # now dual-use
    df["SMKDUAL"] = df["SMKDUAL"].mask((df["SMKECIGST_A"] == 1) & ((df["SMKCIGST_A"] == 1) | (df["SMKCIGST_A"] == 2)), 1)
    return df

def drop_exclusive(df) -> pd.DataFrame:
    # SMKDUAL=0 何も吸っていない人とどちらか吸っている人
    # どちらか吸っている人を除外
    print(len(df["SMKDUAL"] == 0))
    df_drop = df[(df["SMKDUAL"] == 0) & ((df["SMKECIGST_A"] == 1) | (df["SMKCIGST_A"] <= 2))]
    print(len(df_drop))
    df = df.drop(index=df_drop.index)
    return df

def delete_related_var(df) -> pd.DataFrame:
    # delete related var
    related_var_list = [
        "SMKNOW_A",
        "CIGNOW_A",
        "ECIGNOW_A",
        "SMKECIGST_A",
        "SMKCIGST_A",
        "HHRESPSA_FLG",
        "SMOKELSCUR_A",
        "WTFA_A",
        "PPSU",
        "PSTRAT",
        "SRVY_YR",
        "INTV_MON",
        "INTV_QRT",
        "HHSTAT_A",
        "IMPNUM_A",
        "RECTYPE",
        "STREV_A",
    ]
    df = df.drop(related_var_list, axis=1)

    return df

def complement_na(df) -> pd.DataFrame:
    # 一律置換
    df = df.fillna(
        {
            "LANGHM_A":9,
            "AFNOW":2,
            "EMPWRKFT1_A":9,
            "EMDOCCUPN2_A":99,
            "EMDOCCUPN1_A":99,
            "EMDINDSTN2_A":99,
            "EMDINDSTN1_A":99,
            "COVER_A":5,
            "EXCHANGE_A":2,
            "PLNWRKR1_A":99,
            "EXCHPR1_A":9,
            "MAXEDUCP_A":99,
            "INCOTHR_A":99,
            "INCRETIRE_A":9,
            "INCWELF_A":9,
            "INCSSISSDI_A":9,
            "INCSSRR_A":9,
            "CEVOLUN2_A":9,
            "EMDWRKCAT1_A":9,
            "EMDSUPER_A":9,
            "EMPHEALINS_A":9,
            "EMPSICKLV_A":9,
            "EVRMARRIED_A":9,
            "HRTESTLAST_A":9,
            "CBALHDINJ_A":9,
            "HRWHISP_A":9,
            "PAITOOTH3M_A":9,
            "PAIAPG3M_A":9,
            "PAIHDFC3M_A":9,
            "PAILLMB3M_A":9,
            "PAIULMB3M_A":9,
            "PAIBACK3M_A":9,
            "PAIAFFM3M_A":9,
            "PAIWKLM3M_A":9,
            "PAIAMNT_A":9,
            "MHRX_A":9,
            "ANXLEVEL_A":9,
            "WRKHLTHFC_A":2,
            "LIVEHEP_A":9,
            "SHTHEPB1_A":9,
            "SHTSHINGL1_A":9,
            "SHOTTYPE2_A":9,
            "CVDVAC1Y1_A":9999,
            "CVDVAC1M1_A":99,
            "SHTCVD19NM1_A":9,
            "SHTFLUY_A":9999,
            "SHTFLUM_A":99,
            "HYSTEV2_A":9,
            "CERVICEV1_A":9,
            "COLOGUARD1_A":9,
            "FITHEV1_A":9,
            "CTCOLEV1_A":9,
            "COLORECTEV_A":9,
            "DIBLAST1_A":9,
            "RXDL12M_A":2,
            "RXLS12M_A":2,
            "RXSK12M_A":2,
            "HITTEST_A":9,
            "HITCOMM_A":9,
            "HITLOOK_A":9,
            "ACCSSHOM_A":1,
            "USPLKIND_A":9,
            "WELLNESS_A":9,
            "LONGCOVD1_A":9,
            "HINOTYR_A":9,
            "PRVSCOV1_A":9,
            "PRDNCOV1_A":9,
            "PRRXCOV1_A":9,
            "PRDEDUC1_A":9,
            "PLN1PAY6_A":9,
            "PLN1PAY5_A":9,
            "PLN1PAY4_A":9,
            "PLN1PAY3_A":9,
            "PLN1PAY2_A":9,
            "PLN1PAY1_A":9,
            "POLHLD1_A":9,
            "STEPS_A":9,
            "WLK13M_A":9,
            "WLK100_A":9,
            "VIMCSURG_A":9,
            "VIMMDEV_A":9,
            "GESDIB_A":9,
            "ASPONOWN_A":2,
            "ASPMEDEV_A":9,
            "PCNTADTWFP_A":8,
            "PCNTADTWKP_A":8,
            "HICOSTR1_A":99998,
            "ANYINJURY_A":9,
            "REPSTRAIN_A":9,
            "PAIFRQ3M_A":9,
            "TRAVEL_A":9,
            "CHOLLAST_A":9,
            "BPLAST_A":9,
            "CVDDIAG_A":9,
            "HEPEV_A":9,
            "INCTCFLG_A":0,
            "CEVOTELC_A":9,
            "CEMMETNG_A":9,
            "CEVOLUN1_A":9,
            "ACCSSINT_A":9,
            "TRANSPOR_A":9,
            "EMPLASTWK_A":9,
            "THERA12M_A":9,
            "EYEEX12M_A":9,
            "SHTCVD191_A":9,
            "DENNG12M_A":9,
            "DENDL12M_A":9,
            "DENPREV_A":9,
            "LSATIS4_A":9,
            "DEPLEVEL_A":9,
            "FWIC12M_A":9,
            "CIGAR30D_A":99,
            "CIGARCUR_A":9,
            "MENTHOLC_A":9
        }
    )

    # 中央値置換
    empdy_median = df["EMPDYSMSS3_A"].median()
    empwk_median = df["EMPWKHRS3_A"].median()
    df = df.fillna(
        {
            "EMPDYSMSS3_A":empdy_median,
            "EMPWKHRS3_A":empwk_median
        }
    )

    return df

def down_sample(df, rand) -> pd.DataFrame:
    dual_len = len(df[df["SMKDUAL"] == 1])

    # down sampling
    df_dropsample = df[df["SMKDUAL"] == 0].sample(n=(len(df) - (dual_len * 2)), random_state=rand)
    df = df.drop(index=df_dropsample.index)
    return df

def randomforest(df) -> list:
    # explanatory variable
    df_ex = df.drop(["SMKDUAL"], axis=1)

    # response variable
    df_res = df.filter(items=["SMKDUAL"], axis=1)

    # delete NaN columns
    df_ex = df_ex.dropna(how="any", axis=1)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(df_ex, df_res, test_size=0.3, random_state=1)

    # random forest
    clf = RandomForestClassifier(n_estimators=493, random_state=1)
    clf.fit(X_train, y_train.values.ravel())

    # predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # result
    labels = X_train.columns
    importances = clf.feature_importances_

    # feature_importances出力
    label_list = []
    importance_list = []
    for i in labels:
        label_list.append(i)
    for i in importances:
        importance_list.append(float(i))

    with open("feature_importances.txt", "w") as f:
        for li in zip(label_list, importance_list):
            f.write(str(li[0]))
            f.write("\t")
            f.write(str(li[1]))
            f.write("\n")

    return [y_train, y_train_pred, y_test, y_test_pred]


def main() -> None:
    (in_dir) = parse_args()
    csv_list = glob.glob(in_dir + "/*.csv")

    # import csv and concat all
    df_concat = pd.DataFrame()
    for l in csv_list:
        d = pd.read_csv(l)
        df_concat = pd.concat([df_concat, d])
    df = df_concat

    # add dual user
    df = add_dual_col(df)
    df = df.reset_index(drop=True)
    
    # non-user
    df = drop_exclusive(df)
    df = df.reset_index(drop=True)

    # delete related var
    df = delete_related_var(df)

    # 列の50%以上が欠損している場合、列ごと削除
    del_thresh = int(len(df) * 0.5)
    df = df.dropna(thresh=del_thresh, axis=1)
    df = df.reset_index(drop=True)

    # 欠損値の補完
    df = complement_na(df)
    df = df.reset_index(drop=True)

    # カテゴリ化
    df["POVRATTC_A"] = pd.cut(df["POVRATTC_A"], list(range(-1, 12, 1)), labels=[1,2,3,4,5,6,7,8,9,10,11,12]).cat.codes
    df["HICOSTR1_A"] = pd.cut(df["HICOSTR1_A"], list(range(-5000, 40000, 5000)) + [40000, 99997, 99998, 99999], labels=[2,3,4,5,6,7,8,9,10,11,1,1], ordered=False).cat.codes
    df["AGEP_A"] = pd.cut(df["AGEP_A"], [17, 30, 40, 60, 80] + [97, 98, 99], labels=[2,3,4,5,6,1,1], ordered=False).cat.codes
    df["EMPDYSMSS3_A"] = pd.cut(df["EMPDYSMSS3_A"], list(range(-10, 130, 10)) + [130, 997, 998, 999], labels=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,1], ordered=False).cat.codes
    df["WEIGHTLBTC_A"] = pd.cut(df["WEIGHTLBTC_A"], list(range(80, 300, 20)) + [300, 996, 997, 998, 999], labels=[2,3,4,5,6,7,8,9,10,11,12,13,1,1,1], ordered=False).cat.codes
    df["EMPWKHRS3_A"] = pd.cut(df["EMPWKHRS3_A"], [0, 11, 22, 33, 44, 55, 66, 77, 88, 94] + [95, 97, 98, 99], labels=[2,3,4,5,6,7,8,9,10,11,12,1,1], ordered=False).cat.codes
    df = df.reset_index(drop=True)

    # 順位前処理
    with open("edit_v1_org.csv") as f:
        reader = csv.reader(f)
        org_list = []
        for row in reader:
            oline = []
            for r in row:
                if r != "":
                    if r.isdigit():
                        oline.append(int(r))
                    else:
                        oline.append(r)
            org_list.append(oline)

    with open("edit_v1_to.csv") as f:
        reader = csv.reader(f)
        to_list = []
        for row in reader:
            tline = []
            for r in row:
                if r != "":
                    if r.isdigit():
                        tline.append(int(r))
                    else:
                        tline.append(r)
            to_list.append(tline)

    for i, o in enumerate(org_list):
        if str(o[0]) in df.columns:
            t = to_list[i]
            d = dict(zip(o[1:],t[1:]))
            df[str(o[0])] = df[str(o[0])].replace(d)
    df = df.reset_index(drop=True)

    df = df.drop("HHX", axis=1)
    df = df.reset_index(drop=True)

    rand_list = [
        213, 455, 510, 603, 700, 703, 710, 782, 899, 1387, 1705, 1845, 1880, 1973, 2043, 2061, 2286, 2333, 2369, 2498,
        2537, 2558, 2572, 2601, 2622, 2642, 2766, 2852, 2890, 2980, 3069, 3080, 3112, 3127, 3197, 3274, 3325, 3397, 3431, 3471,
        3490, 3512, 3580, 4160, 4211, 4366, 4491, 4562, 4850, 4866, 5071, 5203, 5372, 5573, 5613, 5744, 5803, 5867, 5917, 6149,
        6371, 6455, 6465, 6515, 6659, 6682, 7066, 7278, 7324, 7354, 7523, 7550, 7642, 7746, 7851, 7953, 7976, 8065, 8123, 8159,
        8174, 8207, 8229, 8263, 8331, 8349, 8375, 8408, 8427, 8451, 8648, 8952, 8988, 9119, 9166, 9245, 9318, 9459, 9697, 9833,
    ]
    
    with open("RandomForest_DUALvsNonuser.txt", "w") as f:

        for i in rand_list:

            # down sampling
            df_downsample = down_sample(df, i)
            
            print("down_sample")
            print(df_downsample.shape[0])
            print(df_downsample.shape[1])

            [y_train, y_train_pred, y_test, y_test_pred] = randomforest(df_downsample)

            # result
            recall_train = recall_score(y_train, y_train_pred)
            recall_test = recall_score(y_test, y_test_pred)
            specificity_train = specificity_score(y_train, y_train_pred)
            specificity_test = specificity_score(y_test, y_test_pred)
            precision_train = precision_score(y_train, y_train_pred)
            precision_test = precision_score(y_test, y_test_pred)
            accuracy_train = accuracy_score(y_train, y_train_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            f1_train = f1_score(y_train, y_train_pred)
            f1_test = f1_score(y_test, y_test_pred)
            auc_train = roc_auc_score(y_train, y_train_pred)
            auc_test = roc_auc_score(y_test, y_test_pred)
            
            # write
            out_list = [
                i,
                recall_train,
                specificity_train,
                precision_train,
                accuracy_train,
                f1_train,
                auc_train,
                recall_test,
                specificity_test,
                precision_test,
                accuracy_test,
                f1_test,
                auc_test,
            ]

            f.write('\t'.join([str(i) for i in out_list]))
            f.write("\n")

    sys.exit()


def specificity_score(y_true, y_pred):
    #  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    if tn == 0: return 0
    return tn / (tn + fp)


if __name__ == "__main__":
    main()
