import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from lifelines import CoxPHFitter

# ---------------------------------------------------------
# [Part 1] 데이터 생성
# ---------------------------------------------------------
np.random.seed(42)
N_SAMPLES = 5000

BETA_HBA1C = 0.18
BETA_CV = 0.015
BETA_TIR = -0.008

# 데이터 생성
hba1c = np.clip(5.0 + np.random.lognormal(mean=0.7, sigma=0.5, size=N_SAMPLES), 4.5, 12.0)
noise_tir = np.random.normal(0, 8, N_SAMPLES)
tir = np.clip(100 - (hba1c - 5.0) * 15 + noise_tir, 0, 100)
noise_cv = np.random.normal(0, 5, N_SAMPLES)
cv = np.clip(20 + (hba1c - 5.0) * 3 + (100 - tir) * 0.2 + noise_cv, 10, 70)

hazard_score = np.exp(BETA_HBA1C * (hba1c - 5.6) + BETA_CV * (cv - 36) + BETA_TIR * (tir - 70))
TIME_HORIZON = 10
baseline_hazard = 0.015

T = np.random.exponential(1 / (baseline_hazard * hazard_score))
E = (T < TIME_HORIZON).astype(int)

df = pd.DataFrame({'T': T, 'E': E, 'HbA1c': hba1c, 'CV': cv, 'TIR': tir, 'Hazard': hazard_score})

# ---------------------------------------------------------
# [Part 2] 통계적 모델 검증 (C-index, AIC)
# ---------------------------------------------------------
print("=" * 60)
print("[Step 1: 통계적 모델 성능 비교]")

# 1. Old Model (HbA1c Only)
cph_old = CoxPHFitter()
cph_old.fit(df[['T', 'E', 'HbA1c']], duration_col='T', event_col='E')
score_old = cph_old.concordance_index_

# 2. New Model (HbA1c + CGM)
cph_new = CoxPHFitter()
cph_new.fit(df[['T', 'E', 'HbA1c', 'CV', 'TIR']], duration_col='T', event_col='E')
score_new = cph_new.concordance_index_

aic_diff = cph_old.AIC_partial_ - cph_new.AIC_partial_

print(f"1. C-index: {score_old:.4f} -> {score_new:.4f} (약 {((score_new - score_old) / score_old) * 100:.1f}% 향상)")
print(f"2. AIC 감소폭: {aic_diff:.0f}")
print("-" * 60)

# [Visualization 1-1] C-index Plot
plt.figure(figsize=(8, 5))
models = ['Model A\n(HbA1c Only)', 'Model B\n(HbA1c + CGM)']
c_scores = [score_old, score_new]
bars_c = plt.bar(models, c_scores, color=['#bdc3c7', '#2ca02c'], width=0.5, edgecolor='black')
plt.ylim(0.5, 1.0)
plt.title('Prediction Accuracy (C-index)', fontsize=14, fontweight='bold')
plt.ylabel('C-index (Higher is Better)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars_c:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# [Visualization 1-2] AIC Plot
plt.figure(figsize=(8, 5))
aic_scores = [cph_old.AIC_partial_, cph_new.AIC_partial_]
bars_a = plt.bar(models, aic_scores, color=['#bdc3c7', '#d62728'], width=0.5, edgecolor='black')
min_aic = min(aic_scores) * 0.99
max_aic = max(aic_scores) * 1.01
plt.ylim(min_aic, max_aic)
plt.title('Model Fit (AIC - Lower is Better)', fontsize=14, fontweight='bold')
plt.ylabel('AIC', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars_a:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{height:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# [Part 3] 등급 산정 & NRI (재분류)
# ---------------------------------------------------------
GRADE_CUTOFFS = [5.7, 6.5, 7.5, 9.0]
new_thresholds = []
for cutoff in GRADE_CUTOFFS:
    ref_tir = max(0, 100 - (cutoff - 5.0) * 15)
    ref_cv = min(70, 20 + (cutoff - 5.0) * 3 + (100 - ref_tir) * 0.2)
    ref_score = np.exp(BETA_HBA1C * (cutoff - 5.6) + BETA_CV * (ref_cv - 36) + BETA_TIR * (ref_tir - 70))
    new_thresholds.append(ref_score)

def get_old_grade(val):
    if val < 5.7: return 1
    if val < 6.5: return 2
    if val < 7.5: return 3
    if val < 9.0: return 4
    return 5

def get_new_grade(val):
    if val < new_thresholds[0]: return 1
    if val < new_thresholds[1]: return 2
    if val < new_thresholds[2]: return 3
    if val < new_thresholds[3]: return 4
    return 5

df['Old_Grade'] = df['HbA1c'].apply(get_old_grade)
df['New_Grade'] = df['Hazard'].apply(get_new_grade)
df['Migration'] = df['New_Grade'] - df['Old_Grade']

# NRI Calculation
n_event = len(df[df['E'] == 1])
p_up_event = len(df[(df['E'] == 1) & (df['Migration'] > 0)]) / n_event
p_down_event = len(df[(df['E'] == 1) & (df['Migration'] < 0)]) / n_event
n_none = len(df[df['E'] == 0])
p_down_none = len(df[(df['E'] == 0) & (df['Migration'] < 0)]) / n_none
p_up_none = len(df[(df['E'] == 0) & (df['Migration'] > 0)]) / n_none
nri = (p_up_event - p_down_event) + (p_down_none - p_up_none)

print("[Step 2: NRI (재분류 개선 지표)]")
print(f"NRI Score: {nri:.3f} ({nri * 100:.1f}%)")
print("-" * 60)

# [Visualization 2] Grade Matrix Heatmap
grade_matrix = pd.crosstab(df['New_Grade'], df['Old_Grade'])
for i in range(1, 6):
    if i not in grade_matrix.index: grade_matrix.loc[i] = 0
    if i not in grade_matrix.columns: grade_matrix[i] = 0
grade_matrix = grade_matrix.sort_index(axis=0).sort_index(axis=1)

fig2 = plt.figure(figsize=(8, 6))
sns.heatmap(grade_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=1, linecolor='gray',
            annot_kws={"size": 14, "weight": "bold"})
plt.title('Grade Reclassification Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Old Grade (HbA1c Only)', fontsize=12, fontweight='bold')
plt.ylabel('New Grade (HbA1c + CGM)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# [Part 4] 경제성 분석
CLAIM_AMOUNT = 30_000_000
GRADE_PREMIUMS_LTV = {
    1: 6_000_000, 2: 8_000_000, 3: 10_000_000, 4: 12_000_000, 5: 0
}
CONVERSION_RATE = 0.5
VARIABLE_COST_RATE = 0.30
FIXED_COST = 50_000_000

profits = []
breakdown = {
    'Loss_Avoidance': 0,  # 사고 손실 방어 (+)
    'New_Revenue': 0,  # 신규 매출 (+)
    'Opportunity_Cost': 0,  # 기회비용 손실 (-)
    'Wrong_Acq_Loss': 0  # 잘못된 유치 (-)
}

for idx, row in df.iterrows():
    old_g = row['Old_Grade']
    new_g = row['New_Grade']
    old_premium = GRADE_PREMIUMS_LTV[old_g] if old_g < 5 else 0
    new_premium = GRADE_PREMIUMS_LTV[new_g] if new_g < 5 else 0

    benefit = 0
    is_reacting = (np.random.rand() < CONVERSION_RATE)  # 50% 확률로 행동

    if row['Migration'] > 0:  # 보험료 인상 -> 이탈
        if is_reacting:
            lost_income = old_premium * (1 - VARIABLE_COST_RATE)
            if row['E'] == 1:
                # 사고날 사람을 내보냄 -> 이득 (손실 회피)
                val = CLAIM_AMOUNT - lost_income
                benefit = val
                breakdown['Loss_Avoidance'] += val
            else:
                # 멀쩡한 사람을 내보냄 -> 손해 (기회비용)
                val = -lost_income
                benefit = val
                breakdown['Opportunity_Cost'] += val
        else:
            benefit = 0

    elif row['Migration'] < 0:  # 보험료 인하 -> 신규 가입
        if is_reacting:
            revenue_gain = new_premium * (1 - VARIABLE_COST_RATE)
            if row['E'] == 0:
                # 건강한 사람 데려옴 -> 이득 (매출 증대)
                val = revenue_gain
                benefit = val
                breakdown['New_Revenue'] += val
            else:
                # 사고 날 사람 데려옴 -> 손해 (잘못된 유치)
                val = revenue_gain - CLAIM_AMOUNT
                benefit = val
                breakdown['Wrong_Acq_Loss'] += val
        else:
            benefit = 0

    profits.append(benefit)

df['Economic_Value'] = profits
total_benefit = df['Economic_Value'].sum()
# ---------------------------------------------------------
# [Part 5] BEP 분석 및 결과
# ---------------------------------------------------------

# 폰트 및 스타일 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시뮬레이션 결과 계산
cgm_prices = np.arange(0, 200000, 100)
universal_profits = [total_benefit - (N_SAMPLES * p) - FIXED_COST for p in cgm_prices]
bep_price_univ = next((p for p, prof in zip(cgm_prices, universal_profits) if prof < 0), 0)

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 6))

# 1. 메인 수익 곡선
ax1.plot(cgm_prices, universal_profits, color='tab:blue', linewidth=3, label='기대 순이익 곡선')
ax1.axhline(0, color='black', linewidth=1.2) # 0원 기준선

# 2. Net Profit 축 단위 표시
def billions(x, pos):
    return f'{x / 1e8:.1f}억'
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(billions))

# 3. BEP 마킹 및 텍스트 수정
BEP_COLOR = '#F08080'

if bep_price_univ > 0:
    ax1.scatter(bep_price_univ, 0, color=BEP_COLOR, s=180, zorder=5, edgecolor='white')
    ax1.text(bep_price_univ + 5000, 1e7,
             f'BEP:\n{bep_price_univ:,.0f} 원',
             color=BEP_COLOR, fontweight='bold', fontsize=14, va='bottom')

# 그래프 디테일 설정
plt.title('CGM 도입 시 보험사 기대 순이익 분석', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('CGM 기기 단가 (원)', fontsize=12)
plt.ylabel('순이익 (단위: 억 원)', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()