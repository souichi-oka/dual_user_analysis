import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import seaborn as sns; sns.set()
import scipy as sp
import csv

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

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

# Chi-squared & cramersV
def chi_squared(df) -> list:
    chi_list = []
    for c in df.columns.values:
        crossed_df = pd.crosstab(df[c], df["SMKDUAL"])
        x2, p, dof, expected = sp.stats.chi2_contingency(crossed_df)
        min_d = min(crossed_df.shape) - 1
        n = len(df[c])
        if min_d == 0 or n == 0:
            v = 0
        else:
            v = np.sqrt(x2/(min_d*n))
        chi_list.append([c, x2, p, dof, expected, v])

    return chi_list

#VIFを計算する
def cal_vif(df):
    vif_df = pd.DataFrame()
    rc, scores = collinearity_check(df)
    vif_df["VIF_Factor"] = [1 / (1 - s) for s in scores]
    vif_df["features"] = df.columns
    vif_df.sort_values("VIF_Factor", inplace=True, ascending=False)
    # 変数削除の際にindex指定するため降順でindexをリセット
    vif_df.reset_index(inplace=True, drop=True)
    return vif_df

# VIF値の上位の変数から削除するループを実装
def drop_vif(df, threshold=10):
    df_copy = df.copy()
    while True:
        print("IN drop_vif")
        df_vif_cal = cal_vif(df_copy)
        max_vif = df_vif_cal["VIF_Factor"].max()
        print(df_vif_cal.iloc[0]["features"])
        print(max_vif)
        if max_vif <= threshold:
            print("lower than threshold")
            break
        else:
            print("more than threshold")
            max_vif_feature = df_vif_cal.iloc[0]["features"]
            df_vif_cal.drop(inplace=True, index=0)
            df_vif_cal.reset_index(inplace=True, drop=True)
            df_copy = df_copy[df_vif_cal["features"]]
    return df_vif_cal

# 指定の変数順で削除するループを実装
def drop_vif_by_varlist(df, threshold=5, var_list=[]):
    df_copy = df.copy()
    while True:
        print("IN drop_vif_by_varlist")
        df_vif_cal = cal_vif(df_copy)
        flag = 0
        for var in var_list:
            var_df = df_vif_cal.reset_index().query('features == @var')
            if var_df.empty:
                continue
            else:
                feat_idx = var_df.index[0]
            vif = df_vif_cal.iloc[feat_idx]["VIF_Factor"]
            print(df_vif_cal.iloc[feat_idx]["features"])
            print(vif)
            if vif <= threshold:
                print("lower than threshold")
                flag = 1
                continue
            else:
                print("more than threshold")
                df_vif_cal.drop(inplace=True, index=feat_idx)
                df_vif_cal.reset_index(inplace=True, drop=True)
                df_copy = df_copy[df_vif_cal["features"]]
                flag = 2
                break
        
        if flag == 1:
            break
    return df_vif_cal


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def collinearity_check(Xc, model=None, alpha=1.0, emph=False):
    if model is None:
        model = Ridge(alpha=alpha)

    m, n = Xc.shape
    if n < 2:
        raise ValueError()

    rc = np.empty((n, n))
    scores = np.empty(n)

    X = np.copy(Xc)
    for i in range(n):
        y = np.copy(X[:,i])
        X[:,i] = 1

        model.fit(X, y)
        y_calc = model.predict(X)

        score = r2_score(y, y_calc)
        if score < 0:
            scores[i] = 0
            rc[i] = 0
        else:
            scores[i] = score
            if emph:
                rc[i] = model.coef_ * score
            else:
                rc[i] = model.coef_

        X[:,i] = y
    
    return rc, scores


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

    # VIF takefuji
    chi_list = chi_squared(df)
    chi_dict = {}
    for chi in chi_list:
        chi_dict[chi[0]] = chi[1]

    del chi_dict["SMKDUAL"]
    chi_sorted = sorted(chi_dict.items(), key=lambda x:x[1])
    chi_list_sorted = [x[0] for x in chi_sorted]

    df_vif_cal = drop_vif_by_varlist(df, 4, chi_list_sorted)
    print(df_vif_cal)

if __name__ == "__main__":
    main()
