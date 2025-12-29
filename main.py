import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from lifelines import CoxPHFitter

# =========================================================
# [Part 1] 데이터 생성 (기존 로직 100% 유지)
# =========================================================
np.random.seed(42)
N_SAMPLES = 5000

# 계수 설정
BETA_HBA1C = 0.18
BETA_CV = 0.015
BETA_TIR = -0.008

# 데이터 생성 (HbA1c, TIR, CV)
hba1c = np.clip(5.0 + np.random.lognormal(mean=0.7, sigma=0.5, size=N_SAMPLES), 4.5, 12.0)
noise_tir = np.random.normal(0, 8, N_SAMPLES)
tir = np.clip(100 - (hba1c - 5.0) * 15 + noise_tir, 0, 100)
noise_cv = np.random.normal(0, 5, N_SAMPLES)
cv = np.clip(20 + (hba1c - 5.0) * 3 + (100 - tir) * 0.2 + noise_cv, 10, 70)

# 위험 점수 및 생존 시간(T), 이벤트(E) 생성
hazard_score = np.exp(BETA_HBA1C * (hba1c - 5.6) + BETA_CV * (cv - 36) + BETA_TIR * (tir - 70))
TIME_HORIZON = 10
baseline_hazard = 0.015

T = np.random.exponential(1 / (baseline_hazard * hazard_score))
E = (T < TIME_HORIZON).astype(int)

df = pd.DataFrame({'T': T, 'E': E, 'HbA1c': hba1c, 'CV': cv, 'TIR': tir, 'Hazard': hazard_score})

# =========================================================
# [Part 2] 통계적 모델 검증 (C-index, AIC)
# =========================================================
print("=" * 60)
print("[Step 1: 통계적 모델 성능 비교]")

cph_old = CoxPHFitter()
cph_old.fit(df[['T', 'E', 'HbA1c']], duration_col='T', event_col='E')
score_old = cph_old.concordance_index_

cph_new = CoxPHFitter()
cph_new.fit(df[['T', 'E', 'HbA1c', 'CV', 'TIR']], duration_col='T', event_col='E')
score_new = cph_new.concordance_index_

aic_diff = cph_old.AIC_partial_ - cph_new.AIC_partial_

print(f"1. C-index: {score_old:.4f} -> {score_new:.4f} (약 {((score_new - score_old) / score_old) * 100:.1f}% 향상)")
print(f"2. AIC 감소폭: {aic_diff:.0f}")
print("-" * 60)

# [Visualization 1] C-index Plot
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

# =========================================================
# [Part 3] 등급 산정 & NRI
# =========================================================
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

# =========================================================
# [Part 4] 경제성 분석 (명목 기준)
# =========================================================
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

# [수정] Part 7 CSM 계산의 일관성을 위해 '누가 반응했는지'를 기록합니다.
# 난수 생성 순서나 결과에 전혀 영향을 주지 않으면서 리스트에만 담습니다.
is_reacting_list = []

for idx, row in df.iterrows():
    old_g = row['Old_Grade']
    new_g = row['New_Grade']
    old_premium = GRADE_PREMIUMS_LTV[old_g] if old_g < 5 else 0
    new_premium = GRADE_PREMIUMS_LTV[new_g] if new_g < 5 else 0

    benefit = 0
    is_reacting = (np.random.rand() < CONVERSION_RATE)  # 50% 확률로 행동
    is_reacting_list.append(is_reacting)  # 기록만 함

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
df['Is_Reacting'] = is_reacting_list  # 데이터프레임에 저장 (Part 7용)
total_benefit = df['Economic_Value'].sum()

# ---------------------------------------------------------
# [Added Output] 경제성 분석 결과 텍스트 상세 출력
# ---------------------------------------------------------
print("=" * 60)
print("[Step 2: 경제성 분석 상세 (항목별 가치 - 명목 기준)]")
print(f"1. 사고 손실 방어 (Loss Avoidance)   : +{breakdown['Loss_Avoidance'] / 1e8:.2f} 억 원")
print(f"2. 신규 매출 창출 (New Revenue)      : +{breakdown['New_Revenue'] / 1e8:.2f} 억 원")
print(f"3. 기회비용 손실 (Opportunity Cost)  : {breakdown['Opportunity_Cost'] / 1e8:.2f} 억 원")
print(f"4. 잘못된 유치 손실 (Wrong Acq Loss) : {breakdown['Wrong_Acq_Loss'] / 1e8:.2f} 억 원")
print("-" * 60)
print(f"▶ 알고리즘 총 가치 (Gross Benefit)   : +{total_benefit / 1e8:.2f} 억 원")
print("=" * 60)

# =========================================================
# [Part 5] BEP 분석 및 결과
# =========================================================

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
ax1.axhline(0, color='black', linewidth=1.2)  # 0원 기준선


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

# =========================================================
# [Part 6] IFRS 17 RA(위험조정) 시뮬레이션
# =========================================================
print("=" * 60)
print("[Step 3: IFRS 17 RA(위험조정) 산출 및 CSM 효과 분석]")

# 설정 변수
N_ITERATIONS = 5000  # 몬테카를로 시뮬레이션 횟수
CONFIDENCE_LEVEL = 0.995  # IFRS 17 신뢰수준 (보통 75~90% 사이 사용)
CLAIM_AMT = 30_000_000  # 사고당 보험금

# 1. 각 모델별 사고 확률 예측 (10년 내 사고 확률)
surv_old = cph_old.predict_survival_function(df[['HbA1c']], times=[10]).T[10].values
surv_new = cph_new.predict_survival_function(df[['HbA1c', 'CV', 'TIR']], times=[10]).T[10].values

prob_old = 1 - surv_old  # 사고 확률 (Old Model)
prob_new = 1 - surv_new  # 사고 확률 (New Model)

# 2. 부트스트래핑을 통한 손실 분포(Loss Distribution) 생성
losses_old = []
losses_new = []

for _ in range(N_ITERATIONS):
    events_old = np.random.binomial(1, prob_old)
    total_loss_old = np.sum(events_old) * CLAIM_AMT
    losses_old.append(total_loss_old)

    events_new = np.random.binomial(1, prob_new)
    total_loss_new = np.sum(events_new) * CLAIM_AMT
    losses_new.append(total_loss_new)

losses_old = np.array(losses_old)
losses_new = np.array(losses_new)

# 3. RA (Risk Adjustment) 계산
ra_old = np.percentile(losses_old, CONFIDENCE_LEVEL * 100) - np.mean(losses_old)
ra_new = np.percentile(losses_new, CONFIDENCE_LEVEL * 100) - np.mean(losses_new)
ra_diff = ra_old - ra_new  # RA 감소량 (명목)

# 결과 출력
print(f"--- IFRS 17 Simulation Result (Confidence Level: {CONFIDENCE_LEVEL * 100}%) ---")
print(f"1. RA (Risk Adjustment, 위험조정 - 명목)")
print(f"   - 기존 모델 (HbA1c) : {ra_old / 1e8:.2f} 억 원")
print(f"   - 제안 모델 (CGM)   : {ra_new / 1e8:.2f} 억 원")
print(f"   - RA 감소 효과      : {ra_diff / 1e8:.2f} 억 원 (▼ {((ra_old - ra_new) / ra_old) * 100:.1f}%)")

# [Visualization 3] Loss Distribution & RA Comparison
plt.figure(figsize=(12, 6))
sns.histplot(losses_old, color='gray', label='Old Model Distribution', kde=True, element="step", alpha=0.3)
sns.histplot(losses_new, color='orange', label='New Model Distribution', kde=True, element="step", alpha=0.5)

plt.axvline(np.mean(losses_old), color='gray', linestyle='--', linewidth=1)
plt.axvline(np.percentile(losses_old, CONFIDENCE_LEVEL * 100), color='gray', linestyle='-', linewidth=2,
            label=f'Old VaR ({CONFIDENCE_LEVEL * 100}%)')
plt.axvline(np.mean(losses_new), color='#d35400', linestyle='--', linewidth=1)
plt.axvline(np.percentile(losses_new, CONFIDENCE_LEVEL * 100), color='#d35400', linestyle='-', linewidth=2,
            label=f'New VaR ({CONFIDENCE_LEVEL * 100}%)')

plt.title(f'IFRS 17 Risk Adjustment Analysis (Confidence Level: {CONFIDENCE_LEVEL * 100}%)', fontsize=16,
          fontweight='bold')
plt.xlabel('Estimated Total Loss Amount (Won)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# [Part 7] IFRS 17 CSM 산출 (PV 적용 추가)
# =========================================================
print("=" * 60)
print("[Step 4: IFRS 17 CSM 산출 (할인율 적용)]")

DISCOUNT_RATE = 0.03  # 3%
CGM_PRICE = 60000  # 6만원
FIXED_COST_VAL = 50000000  # 5천만원

delta_pv_premium = 0
delta_pv_claim_benefit = 0  # 보험금 절감 이익 (양수)

# 1. 보험료 및 보험금 PV 계산 Loop
# Part 4에서 저장한 'Is_Reacting' 정보를 그대로 사용하여 정합성 유지
for idx, row in df.iterrows():
    # 할인 기간 (T)
    t = row['T'] if row['T'] < TIME_HORIZON else TIME_HORIZON

    # [보험금 PV Factor] (1+r)^-t (일시금)
    pv_factor_claim = 1 / ((1 + DISCOUNT_RATE) ** t)
    pv_claim = CLAIM_AMOUNT * pv_factor_claim

    # [보험료 PV Factor] 연금 현가 계수 (매년 납입)
    if t > 0:
        annuity_factor = (1 - (1 + DISCOUNT_RATE) ** (-t)) / DISCOUNT_RATE
    else:
        annuity_factor = 0

    old_prem_annual = (GRADE_PREMIUMS_LTV[row['Old_Grade']] / 10) if row['Old_Grade'] < 5 else 0
    new_prem_annual = (GRADE_PREMIUMS_LTV[row['New_Grade']] / 10) if row['New_Grade'] < 5 else 0

    pv_prem_old = old_prem_annual * annuity_factor
    pv_prem_new = new_prem_annual * annuity_factor

    # Migration 로직 (저장된 반응 여부 사용)
    migration = row['Migration']
    is_reacting = row['Is_Reacting']
    is_event = (row['E'] == 1)

    if migration > 0:  # 이탈
        if is_reacting:
            delta_pv_premium -= pv_prem_old  # 보험료 수입 감소 (손해)
            if is_event:
                delta_pv_claim_benefit += pv_claim  # 보험금 지급 회피 (이익)

    elif migration < 0:  # 유입
        if is_reacting:
            delta_pv_premium += pv_prem_new  # 보험료 수입 증가 (이익)
            if is_event:
                delta_pv_claim_benefit -= pv_claim  # 보험금 지급 발생 (손해)

# 2. RA(PV) 계산
# 위에서 구한 Nominal RA Difference(ra_diff)에 할인율 적용
# 평균 듀레이션 5년 가정 (약 0.86배)
pv_factor_ra = 1 / ((1 + DISCOUNT_RATE) ** 5)
delta_pv_ra_benefit = ra_diff * pv_factor_ra

# 3. 비용(PV) 계산
# CGM 비용: 1년차 평균(0.5년) 할인 적용
pv_factor_cgm = 1 / ((1 + DISCOUNT_RATE) ** 0.5)
total_cgm_cost_pv = (N_SAMPLES * CGM_PRICE) * pv_factor_cgm
# 고정비: 현재 가치 그대로 (t=0)
total_fixed_cost_pv = FIXED_COST_VAL

total_expense_pv = total_cgm_cost_pv + total_fixed_cost_pv

# 4. 최종 CSM 변동 계산
total_csm_impact = delta_pv_premium + delta_pv_claim_benefit + delta_pv_ra_benefit - total_expense_pv

print(f"1. 보험료 수입 변동 (PV) : {delta_pv_premium / 1e8:+.2f} 억 원")
print(f"2. 보험금 절감 효과 (PV) : {delta_pv_claim_benefit / 1e8:+.2f} 억 원")
print(f"3. 사업비 증가 (PV)      : -{total_expense_pv / 1e8:.2f} 억 원")
print(f"   (CGM: {total_cgm_cost_pv / 1e8:.2f}억 + 고정비: {total_fixed_cost_pv / 1e8:.2f}억)")
print(f"4. RA 감소 효과 (PV)     : +{delta_pv_ra_benefit / 1e8:.2f} 억 원 (Nominal {ra_diff / 1e8:.2f}억 * 할인)")
print("-" * 40)
print(f"▶ 총 CSM(미래 이익) 증대 : +{total_csm_impact / 1e8:.2f} 억 원")

# [Visualization 5] CSM Waterfall
cats = ['Premium\n(PV)', 'Claims\n(Savings PV)', 'Expenses\n(Cost PV)', 'RA\n(Reduction PV)', 'Total CSM\nIncrease']
vals = [delta_pv_premium, delta_pv_claim_benefit, -total_expense_pv, delta_pv_ra_benefit, total_csm_impact]
cols = ['#1f77b4', '#2ca02c', '#d62728', '#2ca02c', 'navy']

starts = [0]
cur = 0
for v in vals[:-1]:
    cur += v
    starts.append(cur)
starts.append(0)

plt.figure(figsize=(10, 6))
for i in range(len(cats)):
    if i == len(cats) - 1:
        plt.bar(cats[i], vals[i], color=cols[i], edgecolor='black', zorder=3)
        plt.text(cats[i], vals[i] + (vals[i] * 0.05), f"+{vals[i] / 1e8:.1f}억", ha='center', va='bottom',
                 fontweight='bold')
    else:
        plt.bar(cats[i], vals[i], bottom=starts[i], color=cols[i], edgecolor='black', zorder=3)
        lbl = f"{vals[i] / 1e8:+.1f}억"
        # 라벨 위치 조정
        offset = 1e8 if vals[i] >= 0 else -2e8
        plt.text(cats[i], starts[i] + vals[i] + offset, lbl, ha='center', fontweight='bold')

for i in range(len(cats) - 1):
    end = starts[i] + vals[i]
    plt.plot([i, i + 1], [end, end], color='gray', linestyle='--')

plt.axhline(0, color='black')
plt.title('Final CSM Impact Analysis (PV Based)')
plt.ylabel('Value (KRW)')
plt.grid(axis='y', alpha=0.5)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x / 1e8:.0f}억'))
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# [Visualization 6] 직관적인 CSM Bridge Chart (가독성 개선)
# ---------------------------------------------------------
import matplotlib.patches as mpatches

# 데이터 준비 (부호 정리)
# 이익 요인 (+)
benefit_claims = delta_pv_claim_benefit  # 보험금 절감
benefit_ra = delta_pv_ra_benefit  # RA 감소
impact_premium = delta_pv_premium  # 보험료 변동 (보통 이익이나 손해일 수 있음)

# 비용 요인 (-)
cost_system = total_expense_pv  # 기기값 + 고정비 (양수로 입력받아 차트에서 뺌)

# 차트용 데이터 리스트
# 구조: [기초, 사고예방효과, 리스크해소, 보험료변동, 시스템비용, 최종증대분]
steps = [0, benefit_claims, benefit_ra, impact_premium, -cost_system, total_csm_impact]
labels = ['기존 모델\n(Baseline)', '사고 방지\n(보험금 절감)', '불확실성 해소\n(RA 감소)',
          '매출 변동\n(보험료)', '투자 비용\n(기기+고정비)', 'Deep Risk\n최종 가치']

# 워터폴 시작점 계산
base_values = [0]
cum_sum = 0
for i in range(1, len(steps) - 1):
    cum_sum += steps[i]
    base_values.append(cum_sum - steps[i] if steps[i] < 0 else cum_sum - steps[i])  # 막대 시작점
base_values.append(0)  # 마지막 Total 바는 0부터 시작

# 색상 설정 (직관적)
# 이익: 파란색/초록색 계열, 비용: 붉은색 계열, 결과: 강조색(골드/네이비)
colors = ['gray', '#2E8B57', '#3CB371', '#1E90FF', '#CD5C5C', '#000080']
# (Gray, SeaGreen, MediumSeaGreen, DodgerBlue, IndianRed, Navy)

plt.figure(figsize=(12, 7))

# 막대 그리기
for i in range(len(steps)):
    # 막대 높이 (절대값)
    h = abs(steps[i])
    # 막대 시작점 (bottom)
    if i == 0 or i == len(steps) - 1:  # 시작과 끝
        bottom = 0
        h = steps[i] if i != 0 else 0.1  # 첫 막대는 0이라 안보이므로 스킵
    else:
        # 증가분이냐 감소분이냐에 따라 bottom 위치 결정
        bottom = base_values[i] if steps[i] >= 0 else base_values[i] + steps[i]

    # 마지막 결과 막대 그리기 (첫번째 0은 제외)
    if i > 0:
        plt.bar(labels[i], h, bottom=bottom, color=colors[i], edgecolor='black', width=0.6, zorder=3)

        # 값 텍스트 표시
        val_text = f"{steps[i] / 1e8:+.1f}억" if i != len(steps) - 1 else f"+{steps[i] / 1e8:.1f}억"

        # 텍스트 위치 (막대 위쪽 또는 아래쪽)
        text_y = bottom + h + (total_csm_impact * 0.02) if steps[i] >= 0 else bottom - (total_csm_impact * 0.05)

        # 마지막 막대는 굵게 강조
        font_w = 'bold' if i == len(steps) - 1 else 'normal'
        font_s = 13 if i == len(steps) - 1 else 11

        plt.text(i, text_y, val_text, ha='center', va='bottom' if steps[i] >= 0 else 'top',
                 fontsize=font_s, fontweight=font_w, color='black')

# 연결선 그리기 (Bridge 효과)
prev_height = 0
for i in range(1, len(steps)):
    # 이전 막대의 끝 높이
    if i == 1:
        start_line = 0
    else:
        start_line = base_values[i - 1] + steps[i - 1] if steps[i - 1] >= 0 else base_values[i - 1]

    # 현재 막대의 시작 높이와 연결
    # 단순화: 누적 합계를 따라가면 됨
    current_height = sum(steps[:i + 1])

    # 선 그리기 (이전 막대 우측 -> 현재 막대 좌측)
    # plt.plot([i-1.3, i-0.7], [prev_height, prev_height], color='gray', linestyle='--', linewidth=1)
    # 복잡한 선보다 단순 누적선 표시
    pass

# 연결선 (Step-wise lines)
running_total = 0
for i in range(1, len(steps) - 1):
    start = running_total
    running_total += steps[i]
    end = running_total
    # 막대 사이를 잇는 선
    plt.plot([i - 0.3, i + 0.7], [end, end], color='grey', linestyle='--', linewidth=1, zorder=1)

# 기준선
plt.axhline(0, color='black', linewidth=1.5)

# 타이틀 및 꾸미기
plt.title('Deep Risk 모델 도입의 경제적 가치 창출 (Net CSM Impact)', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('가치 변동 (단위: 억 원)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Y축 포맷
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x / 1e8:.0f}억'))

# 범례 추가 (설명 보강)
handles = [
    mpatches.Patch(color='#2E8B57', label='가치 창출 (보험금 절감)'),
    mpatches.Patch(color='#3CB371', label='가치 창출 (불확실성 해소)'),
    mpatches.Patch(color='#CD5C5C', label='비용 발생 (투자)'),
    mpatches.Patch(color='#000080', label='최종 순이익 (Net CSM)')
]
plt.legend(handles=handles, loc='upper left')

plt.tight_layout()
plt.show()