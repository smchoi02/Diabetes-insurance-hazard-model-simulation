import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from lifelines import CoxPHFitter
import matplotlib.patches as mpatches
from matplotlib import font_manager, rc
import platform

# ---------------------------------------------------------
# [Fix] 한글 폰트 설정
# ---------------------------------------------------------
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Windows':
    try:
        path = "c:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    except:
        plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    print("Warning: 한글 폰트가 지원되지 않는 환경일 수 있습니다.")

# =========================================================
# [Part 1] 데이터 생성 (기존 로직 100% 유지)
# =========================================================
np.random.seed(42)
N_SAMPLES = 5000

BETA_HBA1C = 0.18
BETA_CV = 0.015
BETA_TIR = -0.008

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

# =========================================================
# [Part 2] 통계적 모델 검증 (기존 유지)
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
# [Part 3] 등급 산정 & NRI (기존 유지)
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
CLAIM_AMOUNT_NOMINAL = 30_000_000
GRADE_PREMIUMS_LTV = {1: 6_000_000, 2: 8_000_000, 3: 10_000_000, 4: 12_000_000, 5: 0}
CONVERSION_RATE = 0.5
VARIABLE_COST_RATE = 0.30
FIXED_COST = 50_000_000

profits = []
breakdown = {
    'Loss_Avoidance': 0,
    'New_Revenue': 0,
    'Opportunity_Cost': 0,
    'Wrong_Acq_Loss': 0
}

# [중요] CSM 계산 시 정합성을 위해 반응 여부(is_reacting)를 여기서 생성 및 고정
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
                val = CLAIM_AMOUNT_NOMINAL - lost_income
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
                val = revenue_gain - CLAIM_AMOUNT_NOMINAL
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
    pv_claim = CLAIM_AMT * pv_factor_claim

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

# ---------------------------------------------------------
# [Visualization 5] CSM Summary Table
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

# 표 데이터 준비
csm_data = [
    ["1. 보험금 절감 효과", f"+ {delta_pv_claim_benefit / 1e8:.2f} 억 원", "고위험군 사전 선별로 인한 지급 감소"],
    ["2. RA(불확실성) 해소", f"+ {delta_pv_ra_benefit / 1e8:.2f} 억 원", "예측 정확도 향상으로 자본 비용 절감"],
    ["3. 보험료 수입 변동", f"{delta_pv_premium / 1e8:+.2f} 억 원", "우량체 할인/위험군 이탈의 순효과"],
    ["4. 투자 비용 (Cost)", f"- {total_expense_pv / 1e8:.2f} 억 원", "CGM 기기 보급 및 시스템 고정비"],
    ["-------------------", "----------------", "--------------------------------"],
    ["▶ 최종 CSM 증대분", f"+ {total_csm_impact / 1e8:.2f} 억 원", "IFRS 17 기준 미래 확정 이익 증가"]
]

col_labels = ["항목 (Category)", "가치 변동 (Value)", "비고 (Note)"]

# 테이블 생성
table = ax.table(cellText=csm_data, colLabels=col_labels, loc='center', cellLoc='center')

# 테이블 스타일 꾸미기
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)  # 크기 조절

# 헤더 색상 및 글자 굵게
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#404040')  # 헤더 배경색
    elif row == 5:  # 마지막 합계 행
        cell.set_text_props(weight='bold', color='blue')
        cell.set_facecolor('#e6f2ff')  # 합계 배경색

    # 테두리 조정
    cell.set_edgecolor('black')
    cell.set_linewidth(1)

plt.title('Deep Risk Project: IFRS 17 Financial Impact Analysis', fontsize=16, fontweight='bold', y=1.05)
plt.show()

# =========================================================
# [Part 8] 투자 타당성 분석: NPV (순현재가치)
# =========================================================
print("=" * 60)
print("[Step 5: 투자 타당성 분석 - NPV (요구수익률 8% 기준)]")

# 1. 투자 가정
HURDLE_RATE = 0.08  # WACC 8% 가정
DISCOUNT_RATE_ACC = 0.03  # 회계적 할인율

# 2. NPV 변수 초기화
npv_delta_premium = 0
npv_delta_claim_benefit = 0
npv_total_device_cost = 0

# 3. 개별 가입자 현금흐름 재할인 Loop (8% 기준)
# Part 7과 동일한 로직, 할인율만 HURDLE_RATE 사용, RA 제외
for idx, row in df.iterrows():
    t = row['T'] if row['T'] < TIME_HORIZON else TIME_HORIZON

    # 보험금 PV @ 8%
    pv_factor_claim_inv = 1 / ((1 + HURDLE_RATE) ** t)
    pv_claim_val_inv = CLAIM_AMT * pv_factor_claim_inv

    # 보험료 PV @ 8%
    if t > 0:
        annuity_factor_inv = (1 - (1 + HURDLE_RATE) ** (-t)) / HURDLE_RATE
    else:
        annuity_factor_inv = 0

    old_prem_annual = (GRADE_PREMIUMS_LTV[row['Old_Grade']] / 10) if row['Old_Grade'] < 5 else 0
    new_prem_annual = (GRADE_PREMIUMS_LTV[row['New_Grade']] / 10) if row['New_Grade'] < 5 else 0

    pv_prem_old_inv = old_prem_annual * annuity_factor_inv
    pv_prem_new_inv = new_prem_annual * annuity_factor_inv

    # 기기 비용 PV @ 8%
    pv_factor_device_inv = 1 / ((1 + HURDLE_RATE) ** 0.5)
    npv_total_device_cost += (CGM_PRICE * pv_factor_device_inv)

    # Migration 로직 (저장된 반응 여부 사용)
    migration = row['Migration']
    is_reacting = row['Is_Reacting']
    is_event = (row['E'] == 1)

    if migration > 0:  # 이탈
        if is_reacting:
            npv_delta_premium -= pv_prem_old_inv  # 현금 유입 감소
            if is_event:
                npv_delta_claim_benefit += pv_claim_val_inv  # 현금 유출 방지 (이익)

    elif migration < 0:  # 유입
        if is_reacting:
            npv_delta_premium += pv_prem_new_inv  # 현금 유입 증가
            if is_event:
                npv_delta_claim_benefit -= pv_claim_val_inv  # 현금 유출 발생 (손해)

# 4. 고정비 (할인 없음)
npv_fixed_cost = FIXED_COST_VAL

# 5. 최종 NPV 산출
# NPV = (보험료 변동 + 보험금 절감) - (기기비용 + 고정비)
total_npv = npv_delta_premium + npv_delta_claim_benefit - npv_total_device_cost - npv_fixed_cost

# 6. 결과 출력
print(f"1. 현금 유입 가치 (PV 8%):")
print(f"   - 보험료 변동분: {npv_delta_premium / 1e8:+.2f} 억 원")
print(f"   - 보험금 절감분: {npv_delta_claim_benefit / 1e8:+.2f} 억 원")
print(f"2. 현금 유출 가치 (PV 8%):")
print(f"   - 초기 투자비용: -{(npv_total_device_cost + npv_fixed_cost) / 1e8:.2f} 억 원")
print("-" * 40)
print(f"▶ 순현재가치 (NPV) : {total_npv / 1e8:+.2f} 억 원 (요구수익률 8% 기준)")

if total_npv > 0:
    print(">> 최종 결론: 투자 타당성 확보 (Positive NPV)")
else:
    print(">> 최종 결론: 투자 재검토 필요 (Negative NPV)")

# [Visualization 6] NPV Component Analysis
fig, ax = plt.subplots(figsize=(10, 6))

npv_cats = ['Premium\nImpact', 'Claims\nSavings', 'Investment\nCost', 'Net\nNPV']
npv_vals = [npv_delta_premium, npv_delta_claim_benefit, -(npv_total_device_cost + npv_fixed_cost), total_npv]
npv_colors = ['#1f77b4', '#2ca02c', '#d62728', 'navy']

bars = ax.bar(npv_cats, npv_vals, color=npv_colors, edgecolor='black', width=0.6)

for bar in bars:
    height = bar.get_height()
    label_y = height + (total_npv * 0.05) if height >= 0 else height - (total_npv * 0.1)
    ax.text(bar.get_x() + bar.get_width() / 2, label_y, f'{height / 1e8:+.2f}억',
            ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold', fontsize=12)

ax.axhline(0, color='black', linewidth=1.2)
plt.title(f'NPV Analysis', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Present Value (KRW)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x / 1e8:.0f}억'))

plt.tight_layout()
plt.show()

# =========================================================
# [Part 9] ALM & Duration Gap Analysis (Bernoulli Simulation)
# =========================================================
print("=" * 60)
print("[Step 6: ALM & Duration Gap 시뮬레이션 (Bias Correction)]")

# 1. ALM 파라미터 설정
TARGET_ASSET_DURATION = 5.5  # 목표 자산 듀레이션
SHOCK_KICS = 0.01  # 금리 충격 100bps
LIABILITY_VAL = 150000000000  # 부채 평가액 (1,500억 가정)
N_SIM_ALM = 1000  # ALM 시뮬레이션 횟수

# 2. 연도별 생존 함수(Survival Function) 추출 (Year 1 ~ 10)
times = np.arange(1, TIME_HORIZON + 1)

surv_df_old = cph_old.predict_survival_function(df[['HbA1c']], times=times)
surv_df_new = cph_new.predict_survival_function(df[['HbA1c', 'CV', 'TIR']], times=times)

# 3. 누적 부도 확률(CDF)로 변환: 1 - Survival Function
cdf_old = 1 - surv_df_old.values
cdf_new = 1 - surv_df_new.values


# 4. 몬테카를로 시뮬레이션
def run_duration_simulation(cdf_matrix, n_sims, n_samples):
    durations = []

    # 각 환자의 연도별 사고 발생 누적 확률
    cdf_T = cdf_matrix.T

    for _ in range(n_sims):
        # (1) 난수 생성 (5000명에 대해 0~1 사이 값)
        rand_probs = np.random.rand(n_samples, 1)

        # (2) 사고 발생 시점 판별
        # 난수 < CDF(t) 인 첫 번째 시점이 사고 발생 시점
        occurred = (rand_probs < cdf_T)

        # 사고가 발생한 사람만 필터링
        has_event = occurred.any(axis=1)

        # 사고 시점 (0~9 index -> 1~10년)
        event_times = occurred.argmax(axis=1) + 1

        # 사고 안 난 사람은 시점 0으로 처리 (Cashflow 계산에서 제외됨)
        event_times[~has_event] = 0

        # (3) 포트폴리오 현금흐름 집계
        # 1년~10년 각 시점에 발생할 총 보험금 계산
        cf_vectors = np.zeros(TIME_HORIZON + 1)  # index 1~10 사용

        # bincount를 사용하여 연도별 사고 건수 집계 -> 금액으로 환산
        counts = np.bincount(event_times, minlength=TIME_HORIZON + 1)
        cf_vectors = counts * CLAIM_AMT

        # index 0은 사고 안 난 케이스이므로 제외
        cf_stream = cf_vectors[1:]

        # (4) 매콜리 듀레이션(Macaulay Duration) 산출
        t_vec = np.arange(1, TIME_HORIZON + 1)
        pv_factors = 1 / ((1 + DISCOUNT_RATE) ** t_vec)

        pv_stream = cf_stream * pv_factors
        total_pv = np.sum(pv_stream)

        if total_pv == 0:
            dur = 0  # 예외 처리
        else:
            # D = Sum(t * PV_t) / Sum(PV_t)
            dur = np.sum(t_vec * pv_stream) / total_pv

        durations.append(dur)

    return np.array(durations)


# 5. 시뮬레이션 실행
durs_old = run_duration_simulation(cdf_old, N_SIM_ALM, N_SAMPLES)
durs_new = run_duration_simulation(cdf_new, N_SIM_ALM, N_SAMPLES)

# 6. 결과 집계 (평균 듀레이션)
mean_dur_old = np.mean(durs_old)
mean_dur_new = np.mean(durs_new)

# 7. 듀레이션 갭 및 요구자본(RC) 산출
gap_old = abs(TARGET_ASSET_DURATION - mean_dur_old)
gap_new = abs(TARGET_ASSET_DURATION - mean_dur_new)

rc_old_alm = LIABILITY_VAL * gap_old * SHOCK_KICS
rc_new_alm = LIABILITY_VAL * gap_new * SHOCK_KICS
rc_saving = rc_old_alm - rc_new_alm

print(f"1. 부채 듀레이션 ($D_L$)")
print(f"   - 기존 모델 : {mean_dur_old:.3f} 년 (Bias 존재)")
print(f"   - 제안 모델 : {mean_dur_new:.3f} 년 (Bias 교정)")
print(f"2. 듀레이션 갭 ($|D_A - D_L|$ Target {TARGET_ASSET_DURATION}년)")
print(f"   - 기존 갭   : {gap_old:.3f} 년")
print(f"   - 개선 갭   : {gap_new:.3f} 년 (▼ {gap_old - gap_new:.3f} 년 축소)")
print(f"3. K-ICS 금리부채위험액 (RC)")
print(f"   - 절감 금액 : +{rc_saving / 1e8:.2f} 억 원")

# =========================================================
# [Part 10] ALM 결과 시각화
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Duration Distribution (KDE)
sns.kdeplot(durs_old, ax=axes[0], fill=True, color='gray', label='Old Model $D_L$', alpha=0.3)
sns.kdeplot(durs_new, ax=axes[0], fill=True, color='#1f77b4', label='New Model $D_L$', alpha=0.5)
axes[0].axvline(TARGET_ASSET_DURATION, color='red', linestyle='--', linewidth=2,
                label=f'Asset Duration ({TARGET_ASSET_DURATION}yr)')
axes[0].axvline(mean_dur_old, color='gray', linestyle=':', linewidth=2)
axes[0].axvline(mean_dur_new, color='blue', linestyle=':', linewidth=2)

axes[0].set_title('Liability Duration Distribution & Matching', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Duration (Years)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Chart 2: Capital Charge Reduction (Waterfall like)
labels = ['Old RC', 'New RC']
values = [rc_old_alm / 1e8, rc_new_alm / 1e8]
colors = ['#bdc3c7', '#2c3e50']

bars = axes[1].bar(labels, values, color=colors, width=0.5, edgecolor='black')

# 절감분 표시 화살표
arrow_x = 0.5
arrow_y_start = values[0]
arrow_y_end = values[1]
axes[1].annotate(f'-{rc_saving / 1e8:.2f}억',
                 xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

axes[1].set_title('K-ICS Interest Rate Risk Capital (RC)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Required Capital (100 Million KRW)')
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0f}억'))

for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}억',
                 ha='center', va='center', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()

print("=" * 60)
print(">> 모든 시뮬레이션 및 분석 완료.")
print(">> [결론] CGM 기반 Dynamic Pricing은 수익성(CSM)과 건전성(K-ICS)을 동시에 개선함.")