""" 25 - 06 - 07 """
from dask.dataframe.shuffle import sort_values

""" Review -
1. 패키지 로드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

2. 데이터 로드
mpg = pd.read_csv(path + 'mpg.csv')

3. 데이터 가공 함수
    1) exam.query('english >= 70')
    2) exam['math']
"""
# --------------------------------------------------------------------------------------------------------------------
### 데이터 로드
path = 'C:/PData/'

import pandas  as pd
exam = pd.read_csv(path + 'exam.csv')
# print(exam)

### 필요한 변수 추출

# exam 에서 nclass 가 1인 경우만 추출
a = exam.query('nclass == 1')
# print(a)

# 수학 점수만 추출
a = exam['math']
# print(a)

# 반, 수학, 영어 추출
a = exam[['nclass', 'math', 'english']]
# print(a)

b = exam[['math']]
# print(b)

# 변수 제거하기
a = exam.drop(columns='math') # 수학 제거, drop함수는 모든 언어에서 제거함수로써 쓰인다.
# print(a)

### pandas 함수 조합

# 수학 점수가 50점 이상인 학생의 id 와 math 변수 추출
a = exam.query('math >= 50')[['id', 'math']].head(10) # 가로로 길면 가독성이 떨어짐
# print(a)

# 위 코드를 가독성을 높은 것

exam.query('math >= 50') \
    [['id', 'math']] \
    .head(10)

### 정렬
a = exam.sort_values('math') # 오름차순(디폴트값, ascending=True인 경우)
# print(a)

b = exam.sort_values('math', ascending=False)
# print(b)

# 여러 정렬 기준 적용
a = exam.sort_values(['nclass' ,'math'])
# print(a)

# 변수별로 정렬 순서를 다르게 지정
a = exam.sort_values(['nclass', 'math'], ascending=[True, False])
# print(a)

### 파생변수 추가

# 총점 추가
a = exam.assign(total = exam['math'] + exam['english'] + exam['science']) # 원본 데이터에 영향을 주지 않음
# print(a)

# assign() 에 np.where() 적용
import numpy as np
exam.assign(test = np.where(exam['science']  >= 60, 'PASS', 'FAIL'))
# print(a)

# 추가한 변수를 바로 pandas 함수에 활용이 가능하다.
a = exam.assign(total = exam['math'] + exam['english'] + exam['science']) \
    .sort_values('total', ascending=False)
# print(a)

### 집단별 groupby로 요약 agg

# math 평균 구하기
a = exam.agg(mean_math = ('math', 'mean')) # mean_math는 내가 직접 지정하는 변수
# print(a)

# 집단별로 요약통계량 구하기
a = exam.groupby('nclass') \
    .agg(mean_math = ('math', 'mean'))
# print(a)

# 변수를 인덱스로 바꾸지 않기
a = exam.groupby('nclass', as_index=False) \
    .agg(mean_math = ('math', 'mean'))
# print(a)

# 여러가지 요약통계량 한 번에 구하기
a = exam.groupby('nclass') \
    .agg(mean_math = ('math', 'mean'),
         sum_math = ('math', 'sum'),
         median_math  = ('math', 'median'),
         n = ('nclass', 'count')) # 학생 수(빈도, 행의 개수)
# print(a)

""" agg()에 자주 사용하는 요약 통계향 함수
1. mean() : 평균
2. std() : 표준편차
3. sum() : 합계
4. median() : 중앙값
5. min(), max() : 최소값 최대값
6. count() : 빈도
"""

# 집단별로 다시 집단 나누기 -> 집단을 나눈 뒤에 하위 집단을 만드는 작업
mpg = pd.read_csv(path + 'mpg.csv')
# 제조 회사 및 구동 방식별 분리하고 도시 연비의 평균 구하기
a = mpg.groupby(['manufacturer', 'drv']) \
    .agg(mean_cty = ('cty', 'mean'))
# print(a)

"""
Q1. mpg 데이터를 활용해서 제조회사별로 'suv' 자동차의 도시 및 고속도로 합산 연비 평균을 구해서,
    내림차순으로 정렬하고, 1~5위까지 출력하기
"""
a = mpg.query('category == "suv"') \
    .assign(total = (mpg['hwy'] + mpg['cty'])/2) \
    .groupby('manufacturer')\
    .agg(mean_tot = ('total', 'mean'))\
    .sort_values('mean_tot', ascending= False)\
    .head(5)
# print(a)
### 데이터 합치기

# 1. 가로로 합치기

# 중간고사 데이터
test1 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'midterm' : [60,80,70,90,75]})

# 기말고사 데이터
test2 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'final' : [70,83,65,95,80]})
# print(test1)
# print(test2)

"""
1. pd.merge()에 결합할 데이터 프레임 명을 나열
2. how = 'left' : 오른쪽에 입력한 데이터 프레임을 왼쪽 데이터 프레임에 결합
3. on : 데이터를 합칠 때 기준으로 삼을 변수명 입력
"""
total = pd.merge(test1, test2, how='left', on='id')
# print(total)

# 다른 데이터를 활용해서 변수 추가 - 매칭
name = pd.DataFrame({'nclass' : [1,2,3,4,5],
                     'teacher' : ['KIM', 'LEE', 'JUNG','PARK','CHOI']})
# print(name)

# nclass 기준으로 합쳐서 exam_new에 할당
exam_new = pd.merge(exam, name, how='left', on='nclass')
# print(exam_new)

# 2. 세로로 합치기
# 오전 면접자 점수 만들기
group_a = pd.DataFrame({'id' : [1,2,3,4,5],
                        'test' : [60,80,70,95,85]})

# 오루 면접자 점수 만들기
group_b = pd.DataFrame({'id' : [6,7,8,9,10],
                        'test' : [70,80,65,95,80]})

group_all = pd.concat([group_a, group_b], ignore_index=True)
# print(group_all)
""" 인덱스 중복 안되도록 새로 부여하려면 pd.concat() 에 ignore_index = True """
# --------------------------------------------------------------------------------------------------------
""" 데이터 정제 - 누란된 값, 이상한 값 정리
1. 결측치(Missing Value)
- 누락된 값, 비어있는 값
- 데이터 수집 과정에서 발생한 오류로 포함될 가능성
- 함수가 적용되지 않거나 분석 결과가 왜곡되는 문제 발생
- 실제 데이터 분석시 결측치 확인, 제거 후 분석
"""

### 결측치 찾기
""" 결측치 임의로 생성
- NumPy 패키지의 np.nan 입력
- 파이썬에서 결측치 표기는 NaN으로 표시
- 불러온 데이터 파일에 결측치가 있으면 자동으로 NaN이 됨.
"""
import pandas as pd
import numpy as np
df = pd.DataFrame({'s' : ['M', 'F',np.nan, 'M','F'],
                   'score' : [5,4,3,4,np.nan]})
# print(df)

# NaN이 있는 상태로 연산하면 출력 결과도 NaN
# print(df['score'] + 1)

# 결측치 확인
# print(pd.isna(df))

# 결측치 빈도 확인
a = pd.isna(df).sum()
# print(a)

### 결측치 제거

# 1. 결측치가 있는 행 제거
a = df.dropna(subset = 'score') # score 결측치 제거
# print(a)

df_nomiss = df.dropna(subset = 'score')
# print(df_nomiss['score'] + 1)

# 2. 여러 변수에 결측치가 없는 데이터 추출
df_nomiss = df.dropna(subset=['score', 's'])
# print(df_nomiss)

# 3. 결측치가 하나라도 있으면 제거
df_nomiss2 = df.dropna()
# print(df_nomiss2)

""" 간편하지만 분석에 사용할 수 있는 다른 데이터까지 제거가 될 수 있음
    분석에 사용할 변수를 직접 지정해서 결측치 제거하는 방법을 권장 """

### 결측치 대체
"""
- 결측치가 적고 데이터 크다면 결측치를 제거하고 분석해도 무방
- 데이터 작고 결측치가 많다면 데이터 손실로 인해 분석 결과 왜곡 발생
- 결측치 대체법을 사용해서 보완
- 결측치 대체법 : 결측치를 제거하는 대신 다른 값을 채워 넣는 방법
    -> 대표값(평균값, 최빈값) 을 구해 일괄 대체
    -> 통계 분석 기법으로 결측치의 예측값을 추정 후 대체
"""
# print(exam)

exam.loc[[2,7,14], ['math']] = np.nan # 2,7,14 행의 math에 NaN 할당
# print(exam)

# 평균 구하기
# print(exam['math'].mean())

# df.fillna() 로 결측치를 평균값으로 대체
exam['math'] = exam['math'].fillna(55) # math가 NaN이면 55로 대체
# print(exam)

"""
이상치(anomaly) : 정상 범위에서 크게 벗어난 값

    - 실제 데이터에 대부분 이상치 들어있음
    - 제거하지 않으면 분석 결과 왜곡되므로 분석 전에 제거 작업 필요
"""

### 이상치 제거 - 존재할 수 없는 값
"""
논리적으로 존재할 수 없는 값이 있을 경우 결측치로 변환 후 제거
ex) 성별 변수에 1(남), 2(여) 외 3, 9 같은 값이 있다면 3, 9 를 NaN으로 변환
"""

df = pd.DataFrame({'s' : [1,2,1,3,2,1],
                   'score' : [5,4,3,4,2,6]})
# print(df)

# 1. 이상치 확인 - 빈도표를 만들어 존재할 수 없는 값이 있는지 확인
a = df['s'].value_counts(sort = False).sort_index()
# print(a)
a = df['score'].value_counts(sort = False).sort_index()
# print(a)

# 2. 결측 처리
df['s'] = np.where(df['s'] == 3, np.nan, df['s'])
# print(df)
df['score'] = np.where(df['score'] > 5, np.nan, df['score'])
# print(df)

# 3. 제거 후 분석
a = df.dropna(subset=['s', 'score'])\
    .groupby('s')\
    .agg(mean_score = ('score','mean'))
# print(a)

### 이상치 제거 - 극단적인 값
"""
극단치(outlier) : 논리적으로 존재할 수 있지만 극단적으로 크거나 작은 값
- 극단치 있으면 분석 결과 왜곡, 제거 후 분석
- 기준 정하기
    1. 논리적 판단 (ex: 성인 몸무게 40~150kg 벗어나면 매우 드물므로 극단치로 간주)
    2. 통계적 기준 (ex: 상하위 0.3% 또는 +-3 표준편차 벗어나면 극단치로 간주
    3. 상자그림(box plot) 을 이용해 중심에서 크게 벗어난 값을 극단치로 간주
"""
### 상자 그림(Box plot) : 데이터의 분포를 상자 모양으로 표현한 그래프
"""
- 중심에서 멀리 떨어진 값을 점으로 표현 : 극단치
- 상자 그림을 이용해서 극단치 기준을 구할 수 있음
"""

# print(mpg)
import seaborn as sns
import matplotlib.pyplot as plt

# sns.boxplot(data=mpg, y='hwy')
# plt.show()

"""
IQR(사분위 범위) : 1사분위수와 3사분위수의 거리
1.5IQR : IQR의 1.5배
"""

### 극단치 기준값 구하기

# 1. 1사분위수, 3사분위수 구하기
pct25 = mpg['hwy'].quantile(.25)
pct75 = mpg['hwy'].quantile(.75)
# print(pct25, pct75)

# 2. IQR 구하기
iqr = pct75 - pct25
# print(iqr)

# 3. 하한, 상한 구하기 - 극단치 기준값
# print('하한:',pct25 - 1.5 * iqr)
# print('상한:',pct75 + 1.5* iqr)

# 4. 극단치 결측 처리 - 4.5~40.5 벗어나면 NaN 부여
mpg['hwy'] = np.where((mpg['hwy'] < 4.5) | (mpg['hwy'] >40.5),
                      np.nan, mpg['hwy'])

a = mpg['hwy'].isna().sum()
# print(a)

# 5. 결측치 제외하고 분석
a = mpg.dropna(subset=['hwy']) \
    .groupby('drv') \
    .agg(mean_hwy = ('hwy', 'mean'))
# print(a)

# ----------------------------------------------------------------------------------------------
""" 그래프 : 데이터를 보기 쉽게 그림으로 표현한 것

- 추세와 경향성이 드러나 데이터의 특징을 쉽게 이해할 수 있다.
- 새로운 패턴 발견, 데이터의 특징을 잘 전달
- 다양한 그래프
    1. 2차원 그래프, 3차원 그래프
    2. 지도 그래프
    3. 네트워크 그래프
    4. 모션 차트
    5. 인터랙티브 그래프 ...
"""

""" 산점도(Scatter Plot)

- 데이터를 x축과 y축에 만나는 점으로 표현한 그래프
- 나이와 소득처럼 연속값으로 된 두 변수의 관계를 표현할 때 사용
"""

import pandas as pd
mpg = pd.read_csv(path + 'mpg.csv')

# x 축은 배기량, y 축은 고속도로 연비를 나타낸 산점도 생성
import seaborn as sns
# sns.scatterplot(data=mpg, x='displ', y='hwy')
# plt.show()

# x 축 범위 3~6 제한
# sns.scatterplot(data=mpg, x='displ', y='hwy') \
#     .set(xlim= [3,6])
# plt.show()

# x 축 범위 3~6, y축 범위 10~30 제한
# sns.scatterplot(data=mpg, x='displ', y='hwy') \
#      .set(xlim= [3,6], ylim = [10,30])
# plt.show()

# 종류별로 표식 색깔 다르게 표현 - 구동방식별
# sns.scatterplot(data=mpg, x='displ',y='hwy',hue='drv')
# plt.show()

""" 그래프 설정 바꾸기 """

import matplotlib.pyplot as plt

# plt.rcParams.update({'figure.dpi' : '150', # 해상도, 기본값 72
#                      'figure.figsize' : [8,6], # 그림 크기, 기본값 [6,4]
#                      'font.size' : '15', # 글자 크기, 기본값 10
#                      'font.family' : 'Malgun Gothic'}) # 폰트, 기본값 san-serif

# 모든 설정 되돌리기
# plt.rcdefaults()

"""
Q2. 미국 지역의 지역별 인구 통계 정보를 담은 midwest.csv 를 이용해 전체 인구와 아시아인 인구간에
어떤 관계가 있는지
    
    x축에는 poptotal(전체인구), y축은 popasian(아시아인 인구)로 된 산점도 만들어 보세요
"""
# midwest.csv
# midwest = pd.read_csv(path + 'midwest.csv')

# 산점도 생성
# sns.scatterplot(data=midwest, x='poptotal', y='popasian') \
#     .set(xlim=(0, 500000), ylim=(0,10000))
# plt.show()

""" 막대 그래프(bar chart)
- 데이터의 크기를 막대로 표현한 그래프
- 성별 소득 차이처럼 집단 간 차이를 표현할 때 많이 사용
"""

### 평균 막대 그래프
# 1. 집단별 평균표 생성

df_mpg = mpg.groupby('drv', as_index=False) \
    .agg(mean_hwy = ('hwy', 'mean'))
# print(df_mpg)

# 2. 그래프 만들기
# sns.barplot(data=df_mpg, x='drv', y='mean_hwy')
# plt.show()

# 3. 크기 순으로 정렬하기

# 3-1. 데이터 프레임 정렬
df_mpg = df_mpg.sort_values('mean_hwy', ascending=False)

# 3-2. 막대 생성
# sns.barplot(data=df_mpg, x='drv', y='mean_hwy')
# plt.show()

### 빈도 막대 그래프 : 값의 빈도(개수)를 막대길이로 표현한 그래프
# 여러 집단의 빈도를 표현할 때

# 1. 집단별 빈도표 생성
# df_mpg = mpg.groupby('drv', as_index=False)\
#     .agg(n=('drv', 'count'))
# print(df_mpg)

# 2. 막대 그래프 생성
# sns.barplot(data=df_mpg, x='drv', y='n')
# plt.show()

# sns.countplot() 으로 빈도 막대 그래프
# sns.countplot(data=mpg, x='drv')
# plt.show()

"""
Q3. mpg 자동차 중에 어떤 자동차 종류(category)가 많은지 알아보려 한다. 
    sns.barplot()을 사용해서
자동차 종류별 빈도를 표현한 막대를 만들어 보세요. 막대는 빈도가 높은 순으로 정렬하세요.
"""
df_mpg = mpg.groupby('category', as_index=False) \
    .agg(n=('category', 'count')) \
    .sort_values('n', ascending=False)
# print(df_mpg)

# sns.barplot(data=df_mpg, x='n', y='category')
# plt.show()

""" 선 그래프(Line Chart) : 데이터를 선으로 표현한 그래프

- 시간에 따라 달라지는 데이터를 표현할 때 자주 사용
    ex) 환율, 주가지수 등 경제 지표가 시간에 따라 변하는 양상

- 시계열 데이터 : 일별 환율처럼 일정 시간 간격을 두고 나열된 데이터
- 시계열 그래프 : 시계열 데이터를 선으로 표현한 그래프
"""

# economics 데이터 로드
econonics = pd.read_csv(path + 'economics.csv')
# print(econonics.head())

# sns.lineplot(data=econonics, x='date', y='unemploy')
# plt.show()
""" x축에 연월일 을 나타낸 문자가 여러번 겹쳐있다. 날짜 추출을 진행 """

### x 축에 연도 표시하기

# 1. 날짜 시간 타입 변수로 만들기
econonics['date2']  = pd.to_datetime(econonics['date'])

# 2. 변수 타입 확인
# print(econonics.info())

# 3. 연도 추출
# print(econonics['date2'].dt.year)

# 4. 월 추출
# print(econonics['date2'].dt.month)

# 5. 일 추출 dt.day

# 6. 연도 변수 추가
econonics['year'] = econonics['date2'].dt.year
# print(econonics.head())

# 7. x 축에 연도 표시
# sns.lineplot(data=econonics, x='year', y='unemploy', errorbar=None) # 신뢰구간 제거
# plt.show()

''' 상자그림 (Box Plot) : 데이터의 분포 또는 퍼져있는 형태를 직사각형 상자 모양으로 표현한 그래프
- 데이터가 어떻게 분포?
- 평균값만 볼 때보다 데이터의 특징을 더 자세히 볼 수 있다.
'''

# 8. 상자 그림
# sns.boxplot(data=mpg, x='drv', y='hwy')
# plt.show()

""" 상자그림 해석
전륜 구동(f)
- 26~29 정도 사이의 좁은 범위에 자동차가 모여있는 뾰족한 형태의 분포를 가지고 있다.
- 수염의 위 아래에 점 표식이 있으므로 연비가 극단적으로 높거나 낮은 자동차들이 있다.

4륜구동
- 17~27 사이에 자동차가 모여있다
- 중앙값이 상자 밑면에 가까우므로 낮은 값 쪽으로 치우친
  형태의 분포
  
후륜 구동(r)
- 17~14 사이에~
- 수염이 짧고 극단치가 없으므로 자동차의 대부분이 사분위
  범위에 해당한다.
"""
# ---------------------------------------------------------------------------------------------------
""" 한국복지패널데이터분석
- 한국보건사회연구원 발간 조사 자료
- 전국 7천여 가구 선정, 2006년부터 매년 추적 조사한 자료
- 경제활동, 생활 실태, 복지욕구 등 천여 개 변수로 구성됨
- 다양한 분야의 연구자, 정책 전문가들이 활용을 함
- 엄밀한 절차로 수집되고 다양한 변수가 있으므로 데이터 분석 연습하기 좋은 데이터
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### 데이터 로드
raw_welfare = pd.read_spss(path + 'Koweps_hpwc14_2019_beta2.sav')

### 복사본 생성
welfare = raw_welfare.copy()

### 데이터 검토
# print(welfare)
# print(welfare.info())

### 활용할 변수 선정 & 변수명 수정
welfare = welfare.rename(columns ={'h14_g3' : 's', # 성별
                                   'h14_g4' : 'birth', # 태어난 연도
                                   'h14_g10' : 'marriage_type', # 혼인 상태
                                   'h14_g11' : 'religion', # 종교
                                   'p1402_8aq1' : 'income', # 월급
                                   'h14_eco9' : 'code_job', # 직업 코드
                                   'h14_reg7' : 'code_region'}) # 지역 코드

"""
- 데이터 분석 절차
1. 변수 검토 및 전처리
    변수 특징 파악, 이상치와 결측치 정제
    변수의 값을 다루기 쉽게 변경
    
    *분석에 활용할 변수 각각 전처리*
2. 변수 간 관계 분석
    데이터 요약표, 그래프 생성
    분석 결과 해석
"""

""" 성별에 따른 월급 차이 - 남 녀 누가 돈을 많이 벌까 ? """

### 성별 변수 검토 및 전처리

# 1. 변수 검토
# print(welfare['s'].dtypes)
# a = welfare['s'].value_counts()
# print(a) -> 이상치가 없음

# 2. 결측치 확인
# a = welfare['s'].isna().sum()
# print(a)

# 3. 성별 항목에 이름을 부여
welfare['s'] = np.where(welfare['s'] == 1, 'male', 'female')
# print(welfare['s'].value_counts())

### 월급 변수 검토 및 전처리
# 1. 변수 검토
# print(welfare['income'].dtypes)
# print(welfare['income'].describe()) # 이상치는 없음

# 2. 전처리
a  = welfare['income'].isna().sum()
# print(a) -> 결측치 존재

### 성별에 따른 월급 차이 분석
# 1. 성별 월급 평균표 생성
s_income = welfare.dropna(subset = 'income')\
    .groupby('s', as_index=False)\
    .agg(mean_income = ('income', 'mean'))
# print(s_income)

# 2. 막대 그래프 생성
# sns.barplot(data=s_income, x='s', y='mean_income', hue='s')
# plt.show()

''' 나이와 월급의 관계 - 몇 살에 돈을 가장 많이 벌까? '''

### 나이 변수 검토 및 전처리
# a = welfare['birth'].dtypes
# print(a)

# print(welfare['birth'].describe())

# sns.histplot(data=welfare, x='birth')
# plt.show()

# a = welfare['birth'].isna().sum()
# print(a) -> 결측치 없음

### 파생변수 - 나이 생성
welfare = welfare.assign(age = 2019 - welfare['birth'] + 1)
# print(welfare['age'].describe())

### 나이와 월급의 관계 분석
# 1. 나이 월급 평균표
age_income = welfare.dropna(subset = 'income') \
    .groupby('age') \
    .agg(mean_income = ('income','mean'))
# print(age_income)

# 2. 시각화
# sns.lineplot(data=age_income, x='age',y='mean_income')
# plt.show()

""" 연령대에 따른 월급 차이 - 어떤 연령대의 월급이 높은가? """

### 파생변수 - 연령대 생성
welfare = welfare.assign(ageg =  np.where(welfare['age'] < 30, 'young',
                                          np.where(welfare['age'] <= 59, 'middle', 'old')))
# print(welfare['ageg'].value_counts())

# sns.countplot(data=welfare, x='ageg')
# plt.show()

### 연령대에 따른 월급 차이
# 연령대 월급 평균표
ageg_income = welfare.dropna(subset = 'income')\
    .groupby('ageg', as_index=False)\
    .agg(mean_income = ('income', 'mean'))
# print(ageg_income)

# 막대 그래프 생성
# sns.barplot(data=ageg_income, x='ageg', y='mean_income')
# plt.show()

# 막대 정렬
sns.barplot(data=ageg_income, x='ageg', y='mean_income',
            order=['young', 'middle', 'old'])
plt.show()

# day6 마침