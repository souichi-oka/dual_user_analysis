import argparse
import pandas as pd
import seaborn as sns
import glob
import seaborn as sns; sns.set()
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

def drop_non_smokeless(df) -> pd.DataFrame:
    # SMKDUAL=0 何も吸っていない人とどちらか吸っている人
    # SLT
    df_drop = df[(df["SMKDUAL"] == 0) & (df["SMOKELSCUR_A"] > 2)]
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

def spearmanr(df) -> list:
    spear_list = []
    for c in df.columns.values:
        res = sp.stats.spearmanr(df[c], df["SMKDUAL"])
        stat = res.statistic
        p = res.pvalue
        spear_list.append([c, stat, p])

    return spear_list

def kendall(df) -> list:
    kendall_list = []
    for c in df.columns.values:
        corr, p = sp.stats.kendalltau(df[c], df["SMKDUAL"])
        kendall_list.append([c, corr, p])

    return kendall_list

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

    vif4_chi_takefuji = [
        "INCSSRR_A",
        "ANXMED_A",
        "OVER65FLG_A",
        "HIKIND10_A",
        "AGEP_A",
        "THERA12M_A",
        "DEPMED_A",
        "DENDL12M_A",
        "SAPARENTSC_A",
        "PCNTLT18TC",
        "INCWRKO_A",
        "MHRX_A",
        "HIKIND05_A",
        "DIBLAST1_A",
        "RXLS12M_A",
        "WLK100_A",
        "INCWELF_A",
        "FDSRUNOUT_A",
        "HINOTYR_A",
        "HOUTENURE_A",
        "TRAVEL_A",
        "FDSBALANCE_A",
        "DIFF_A",
        "STEPS_A",
        "PRDEDUC1_A",
        "RXSK12M_A",
        "ANXFREQ_A",
        "SOCERRNDS_A",
        "EMPSICKLV_A",
        "HISDETP_A",
        "MEDNG12M_A",
        "AFVET_A",
        "MEDDL12M_A",
        "NATUSBORN_A",
        "SHTFLU12M_A",
        "PAIFRQ3M_A",
        "ORIENT_A",
        "DISAB3_A",
        "FSNAP12M_A",
        "PRVSCOV1_A",
        "DEPEV_A",
        "DEPFREQ_A",
        "SOCSCLPAR_A",
        "RXDL12M_A",
        "POVRATTC_A",
        "ANXLEVEL_A",
        "WLK13M_A",
        "INCSSISSDI_A",
        "INCRETIRE_A",
        "HIKIND07_A",
        "CEVOTELC_A",
        "HISPALLP_A",
        "SINCOVDE_A",
        "CIGAREV_A",
        "HOUYRSLIV_A",
        "ECIGEV_A",
        "ANXEV_A",
        "SINCOVVS_A",
        "DIBEV_A",
        "SHTFLUM_A",
        "MARSTAT_A",
        "SCHCURENR_A",
        "UPPSLFCR_A",
        "RXDG12M_A",
        "PIPEEV_A",
        "MHTHND_A",
        "RX12M_A",
        "MHTHRPY_A",
        "MEDICAID_A",
        "SMOKELSEV_A",
        "RACEALLP_A",
        "LASTDR_A",
        "PHSTAT_A",
        "HOMEHC12M_A",
        "INCINTER_A",
        "OTHGOV_A",
        "MILITARY_A",
        "HIKIND09_A",
        "SMKEV_A",
        "SHTPNUEV_A",
        "EDUCP_A",
        "SOCWRKLIM_A",
        "UPPRAISE_A",
        "HIKIND08_A",
        "PCNTADLT_A",
        "HYPEV_A",
        "UPPOBJCT_A",
        "COGMEMDFF_A",
        "ARTHEV_A",
        "OTHPUB_A",
        "GESDIB_A",
        "PAYWORRY_A",
        "HOSPONGT_A",
        "CHDEV_A",
        "EMERG12MTC_A",
        "USPLKIND_A",
        "PREDIB_A",
        "HIKIND03_A",
        "HICOSTR1_A",
        "CHLEV_A",
        "MIEV_A",
        "WORKHEALTH_A",
        "COMDIFF_A",
        "AVAIL_A",
        "VIRAPP12M_A",
        "HEARINGDF_A",
        "PAYBLL12M_A",
        "HLTHCOND_A",
        "MEDRXTRT_A",
        "DEMENEV_A",
        "WELLNESS_A",
        "INCTCFLG_A",
        "VISIONDF_A",
        "EXCHANGE_A",
        "URBRRL",
        "SINCOVRX_A",
        "SMKDUAL",
        "HEARAID_A",
        "BMICAT_A",
        "WEARGLSS_A",
        "NUMCAN_A",
        "COPDEV_A",
        "ANGEV_A",
        "EMPDYSMSS3_A",
        "HEIGHTTC_A",
        "URGNT12MTC_A",
        "AFNOW",
        "IMPINCFLG_A",
        "ASEV_A",
        "EMPWKHRS3_A",
        "REGION",
        "CFSEV_A",
        "MLTFAMFLG_A",
        "EPIEV_A",
        "WRKHLTHFC_A",

    ]
    df = df.filter(items=vif4_chi_takefuji)
    df = df.reset_index(drop=True)

    spear_list = spearmanr(df)
    kendall_list = kendall(df)

    with open("spear_list.txt", "w") as f:
        for s in spear_list:
            f.write('\t'.join([str(i) for i in s[:3]]))
            f.write("\n")

    with open("kendall_list.txt", "w") as f:
        for s in kendall_list:
            f.write('\t'.join([str(i) for i in s[:3]]))
            f.write("\n")


if __name__ == "__main__":
    main()
