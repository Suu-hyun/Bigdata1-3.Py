""" 25 - 06 - 14 """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fontTools.otlLib.optimize import compact
from fontTools.subset import subset

from day6 import econonics

path = 'C:/PData/' # 파일 경로

### 데이터 로드
raw_welfare = pd.read_spss(path + 'Koweps_hpwc14_2019_beta2.sav')

### 복사본 생성
welfare = raw_welfare.copy()

### 데이터 검토
# print(welfare)
# print(welfare.info())

### 활용할 변수 선정 & 변수명 수정
welfare = welfare.rename(columns={'h14_g3': 's',  # 성별
                                  'h14_g4': 'birth',  # 태어난 연도
                                  'h14_g10': 'marriage_type',  # 혼인 상태
                                  'h14_g11': 'religion',  # 종교
                                  'p1402_8aq1': 'income',  # 월급
                                  'h14_eco9': 'code_job',  # 직업 코드
                                  'h14_reg7': 'code_region'})  # 지역 코드

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
a = welfare['income'].isna().sum()
# print(a) -> 결측치 존재

### 성별에 따른 월급 차이 분석
# 1. 성별 월급 평균표 생성
s_income = welfare.dropna(subset='income') \
    .groupby('s', as_index=False) \
    .agg(mean_income=('income', 'mean'))
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
welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
# print(welfare['age'].describe())

### 나이와 월급의 관계 분석
# 1. 나이 월급 평균표
age_income = welfare.dropna(subset='income') \
    .groupby('age') \
    .agg(mean_income=('income', 'mean'))
# print(age_income)

# 2. 시각화
# sns.lineplot(data=age_income, x='age',y='mean_income')
# plt.show()



""" 연령대에 따른 월급 차이 - 어떤 연령대의 월급이 높은가? """

### 파생변수 - 연령대 생성
welfare = welfare.assign(ageg=np.where(welfare['age'] < 30, 'young',
                                       np.where(welfare['age'] <= 59, 'middle', 'old')))
# print(welfare['ageg'].value_counts())

# sns.countplot(data=welfare, x='ageg')
# plt.show()

### 연령대에 따른 월급 차이
# 연령대 월급 평균표
ageg_income = welfare.dropna(subset='income') \
    .groupby('ageg', as_index=False) \
    .agg(mean_income=('income', 'mean'))
# print(ageg_income)

# 막대 그래프 생성
# sns.barplot(data=ageg_income, x='ageg', y='mean_income')
# plt.show()

# 막대 정렬
# sns.barplot(data=ageg_income, x='ageg', y='mean_income',
#             order=['young', 'middle', 'old'])
# plt.show()

''' 연령대 및 성별 월급 차이 - 성별 월급 차이는 연령대 별로 다를까? '''
s_income = welfare.dropna(subset='income') \
    .groupby(['ageg', 's'], as_index=False) \
    .agg(mean_income = ('income', 'mean'))
# print(s_income)


# 막대 그래프 생성
# sns.barplot(data=s_income, x='ageg', y='mean_income', hue='s',
#             order=['young','middle','old'])
# plt.show()

### 나이 및 성별 월급 차이
s_age = welfare.dropna(subset='income') \
    .groupby(['age', 's'], as_index=False) \
    .agg(mean_income = ('income', 'mean'))
# print(s_age)

# 선 그래프 생성
# sns.lineplot(data=s_age, x='age', y='mean_income', hue='s')
# plt.show()

''' 직업별 월급 차이 - 어떤 직업이 돈을 가장 많이 버는가? '''

# print(welfare['code_job'].dtypes)
# print(welfare['code_job'].value_counts())

# 전처리 - 코드분에 직종코드 로드
list_job = pd.read_excel(path + 'Koweps_Codebook_2019.xlsx', sheet_name = '직종코드')
# print(list_job.head())
# print(list_job.shape)

### welfare 에 list_job 결합
welfare = welfare.merge(list_job, how='left', on = 'code_job')

# code_job 에 결측치 제거하고 code_job, job 출력
a = welfare.dropna(subset=['code_job'])[['code_job', 'job']].head()
# print(a)

# 직업별 월급 평균표 생성
job_income = welfare.dropna(subset = ['job','income']) \
    .groupby('job',as_index=False) \
    .agg(mean_income = ('income','mean'))
# print(job_income)

# 월급이 많은 직업 정렬 - 상위 10개 추출
top10 = job_income.sort_values('mean_income',ascending=False).head(10)
# print(top10)

# 맑은 도딕 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Malgun Gothic'})

# 막대그래프 생성
# sns.barplot(data=top10, x='mean_income', y='job', hue='job')
# plt.show()

''' 성별 직업 빈도 - 성별에 따라 어떤 직업이 가장 많은가 ? '''
# 남성 직업 빈도 상위 10개
job_male = welfare.dropna(subset=['job']) \
    .query('s =="male"') \
    .groupby('job', as_index=False) \
    .agg(n = ('job', 'count')) \
    .sort_values('n', ascending=False) \
    .head(10)
# print(job_male)

# 시각화
# sns.barplot(data=job_male, x='n', y='job').set(xlim=[0,500])
# plt.show()

# 여성 빈도 상위 10개
job_female = welfare.dropna(subset=['job']) \
    .query('s =="female"') \
    .groupby('job', as_index=False) \
    .agg(n = ('job', 'count')) \
    .sort_values('n', ascending=False) \
    .head(10)

# 시각화
# sns.barplot(data=job_female, y='job', x='n').set(xlim=[0,500])
# plt.show()

''' 지역별 연령대 비율 - 지역별로 노년층이 가장 많은 지역이 어딜까 ? '''

# 지역변수 검토 및 전처리
# print(welfare['code_region'].dtypes) -> float64
# print(welfare['code_region'].value_counts())

# 지역 코드 목록 생성 - 숫자에 지역명 붙이기
list_region = pd.DataFrame({'code_region' : [1,2,3,4,5,6,7],
                            'region' : ['서울',
                                         '수도권(인천/경기)',
                                         '부산/경남/울산',
                                         '대구/경북',
                                         '대전/충남',
                                         '강원/충북',
                                         '광주/전남/전북/제주도']})

# print(list_region)

# 지역명 변수 매칭
welfare = welfare.merge(list_region, how = 'left', on='code_region')
# print(welfare[['code_region', 'region']]),head(20)

### 지역별 연령대 비율 분석
# 1. 지역별 연령대 비율표
region_ageg = welfare.groupby('region', as_index=False) \
    ['ageg']\
    .value_counts(normalize=True) # 비율 옵션 추가
# print(region_ageg)

# 2. 백분율로 바꾸기
region_ageg = region_ageg.assign(proportion = region_ageg['proportion']*100) \
    .round(1) # 반올림
# print(region_ageg)

# 3. 막대그래프 생성
# sns.barplot(data=region_ageg, y='region', x='proportion', hue='ageg')
# plt.show()

# 4. 누적 비율 막대 그래프 생성
# 4-1. 피벗하기
# pivot_df = region_ageg[['region','ageg','proportion']].pivot(index = 'region',
#                                                               columns = 'ageg', # ageg는 agegroup약자임
#                                                               values = 'proportion')
# print(pivot_df)

# 4-2. 가로 막대 그래프 생성
# pivot_df.plot.barh(stacked = True)
# plt.show()

# 4-3. 노년층 비율 기준 정렬, 변수 순서 바꾸기
# reorder_df = pivot_df.sort_values('old', ascending=False)[['young','middle','old']] # ascending=을 쓰면 작은 순으로 old 정렬이 가능, 없으면 큰 순
# print(reorder_df)

# 4-4. 누적 가로 막대 다시 만들기
# reorder_df.plot.barh(stacked=True)
# plt.show()

# --------------------------------------------------------------------------------------------------------------------------

''' 통계적 가설 검정 

부산시에 관광객이 가장 많은 구는 해운대구다.
-> 입증 ( 내 말이 맞다 or 내 말이 우연일 수 도 있다. 이걸을 고려해라)

1. 기술 통계 : 데이터를 요약해서 설명하는 통계 분석 기법
2. 추론 통계 : 어떤 값이 발생할 확률을 계산하는 통계 분석 기법

ex) 성별에 따른 월급 차이가 우연히 발생할 확률을 계산

- 이런 차이가 우연히 나타날 확률이 작다면
    -> 성별에 따른 월급 차이가 통계적으로 유의하다고 결론
- 이런 차이가 우연히 나타날 확률이 크다면
    -> 성별에 따른 월급 차이가 통계적으로 유의하지 않다고 결론

- 기술 통계 분석에서 집단 간 차이가 있는 것으로 나타났더라도 이는 우연에 의한 차이일 수 도 있음
- 신뢰할 수 있는 결론을 내리려면 유의확률을 계산하는 통계적 가설 검정 절차는 거쳐야 한다.

# 통계적 가설 검정
- 유의확률을 사용해서 가설을 검정하는 방법
- 유의확률(p-value) : 실제로는 집단 간에 차이가 없는데 우연히 차이가 있는 데이터가 추출될 확률 - 기준 0.05

1. 유의확률이 기준보다 크면
    - 집단 간 차이가 통계적으로 유의하지 않다
    - 실제로 차이가 없더라도 우연에 의해 이런 정도의 차이가 관찰될 가능성이 크다는 의미
2. 유의확률이 기준보다 작다면
    - 집단 간 차이가 통계적으로 유의하다
    - 실제로 차이가 없는데 우연히 이런 정도의 차이가 관찰될 가능성이 작다, -> 우연으로 보기 힘들다.
'''

''' t검정(t-test) : 두 집단의 평균에 통계적으로 유의한 차이가 있는지 알아볼 때 사용하는 통계 분석 기법 '''

### compact 자동차와 suv 자동차의 도시 연비 t 검정

# 1. 기술 통계 진행
mpg=pd.read_csv(path + 'mpg.csv')

a = mpg.query('category in ["compact","suv"]') \
    .groupby('category', as_index=False) \
    .agg(n = ('category', 'count'),
         mean = ('cty','mean'))
# print(a)
''' t-검정
1. 비교하는 집단의 분산(값이 퍼져 있는 정도)이 같은지 여부에 따라 적용하는 공식이 다름
2. equal_var = True : 집단 간 분산이 같다고 가정
'''

compact = mpg.query('category ==  "compact"')['cty']
suv = mpg.query('category ==  "suv"')['cty']

# t-test
from scipy import stats
t = stats.ttest_ind(compact, suv, equal_var=True)
# print(t)
''' 해석
p-value 가 0.05 미만이면 ' 집단간 차이가 통계적으로 유의하다'
실제로는 차이가 없는데 이런 정도의 차이가 우연히 관찰될 확률이 5% 보다 작다면, 우연으로 보기 힘들다

pvalue=2.3909550904711282e-21 : 2.390... * 10의 -21승

-> p-value 가 0.05보다 작다고 나왔기 때문에 'compact와 suv 간 도시 연비 차이가 통계적으로 유의하다'
'''

'''
Q1. 유종(f1)중에 일반 휘발유(r) 와 고급 휘발유(p) 의 도시 연비 t검정
'''
r = mpg.query('fl ==  "r"')['cty']
p = mpg.query('fl ==  "p"')['cty']

# t-test
t = stats.ttest_ind(r, p, equal_var=True)
# print(t)
'''
p-value 가 0.2875 로 0.05 보다 크다
실제로는 차이가 없는데 우연히 이런 정도의 차이가 관찰될 확률 28.75%
-> 일반 휘발유와 고급 휘발유를 사용하는 자동타의 도시 연비 차이가 통계적으로 유의하지 않다.
'''

''' 상관분석(Correlation analysis)
    : 두 연속 변수가 서로 관련이 있는지 검정하는 통계 분석 기법

- 상관계수
    -> 두 변수가 얼마나 관련되어 있는지, 관련성의 정도를 파악할 수 있다.
    -> 0~1 사이의 값, 1에 가까울수록 관련성이 크다는 의미
    -> 양수면 정비례, 음수면 반비례 관계
'''

### economics 데이터에서 실업자수 와 개인소비지출의 상관관계
economics = pd.read_csv(path + 'economics.csv')

c = economics[['unemploy', 'pce']].corr()
# print(c)

# 2. 유의확률 구하기 - 상관분석
c = stats.pearsonr(economics['unemploy'], economics['pce'])
# print(c)
'''
statistic=0.6145176141932082 : 상관계수 -> 양수 0.61이 나왔으므로 정비례 관계
pvalue=6.773527303289964e-61 : 유의 확률이 0.05 미만이므로 실업자 수 와 개인 소비지출의 상관관계가 통계적으로
                                유의하다.
'''

### 상관행렬 히트맵
'''
상관행렬
- 모든 변수의 상관관계를 나타낸 행렬
- 여러 변수의 관련성을 한꺼번에 알아보고 싶을 때 사용
- 어떤 변수끼리 관련이 크고 적은지 한 눈에 파악할 수 있음
'''
mtcars = pd.read_csv(path + 'mtcars.csv')
# print(mtcars)

car_cor = mtcars.corr() # 상관 행렬 만들기
# print(car_cor)
car_cor = round(car_cor, 2) # 소수점 둘째 자리까지 반올림
# print(car_cor)

# 히트맵 : 값의 크기를 색깔로 표현한 그래프
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi' : '120',
                     'figure.figsize' : [7.5,5.5]})

import seaborn as sns
# sns.heatmap(car_cor,
#             annot=True, # 상관계수 표시 여부
#             cmap='RdBu') # 컬러맵
# plt.show()

# 대각 행렬 제거 - 히트맵의 대각선 기준으로 왼쪽 아래와 오른쪽 위의 값이 대칭해서 중복됨.
''' mask : 배경 '''
# 1. mask 만들기 = 상관 행렬의 행과 열의 수만큼 0으로 채운 배열(array)을 만듬.
import numpy as np
mask = np.zeros_like(car_cor)
# print(mask)

# mask의 오른쪽 위 대각 행렬을 1로 바꿈
mask[np.triu_indices_from(mask)] = 1
# print(mask)

# 2. 히트맵에 mask 적용
# sns.heatmap(data=car_cor,
#             annot=True,
#             cmap='RdBu',
#             mask=mask)
# plt.show()

# 3. 빈 행과 열 지우고 보기 좋게 수정
mask_new = mask[1:, :-1] # mask 첫 번째 행, 마지막 열 제거
cor_new = car_cor.iloc[1:, :-1] # 상관행렬의 첫 번째 행, 마지막 열 제거

# sns.heatmap(data=cor_new,
#             annot=True,
#             cmap='RdBu',
#             mask = mask_new)
# plt.show()

# 4. 히트맵 옵션
# sns.heatmap(data=cor_new,
#             annot = True, # 상관계수 표시 여부
#             cmap = 'RdBu', # 컬러맵
#             mask = mask_new, # 마스크
#             linewidths=0.5, # 경계 구분선 추가
#             vmax=1, # 가장 진한 파란색으로 표현할 최대값
#             vmin=-1, # 가장 진한 빨간색으로 표현할 최소값
#             cbar_kws = {'shrink' : .5}) # 범례 크기 줄이기, .5는 0.5같은 말
# plt.show()

# -------------------------------------------------------------------------------------------------------------------
''' 텍스트 마이닝 : 문자로 된 데이터에 가치 있는 정보를 얻어내는 분석 기법

- 형태소 분석 : 문장을 구성하는 어절들이 어떤 품사인지 파악하는 과정

- KoNLPy 패키지 : 한글 텍스트 형태소 분석 패키지

1. 운영 체제 버전에 맞는 JAVA 설치, 환경 변수 편집
2. 의존성 패키지 설치 -> pip install jpype1
3. pip install konlpy
'''

### 연설문 로드
'''
파이썬에서 텍스트 파일을 읽어올 때 open() 함수를 쓰게 된다.

인코딩 : 컴퓨터가 문자를 표현하는 방식, 문서마다 인코딩 방식이 다르기 때문에 문서 파일과 
        프로그램의 인코딩이 맞지 않으면 문자가 깨지게 된다.
'''
moon = open(path + 'speech_moon.txt', encoding='UTF-8').read()
# print(moon)

### 가장 많이 사용된 단어 확인

# 1. 불필요한 문자 제거
''' re : 문자 처리 패키지 '''
# jvm 파일을 환경변수편집으로 경로 설정, javahome파일을 만들어준다 여기에 넣고 위에서 두번째 까지 올려주면 됨, chatgpt 활용
import re
'''
정규표현식 : 특정한 규칙을 가진 문자열을 표현하는 언어
[^가-힣] : 한글이 아닌 모든 문자라는 뜻을 가진 정규표현식
[A-Z,a-z] : 모든 언어
'''
moon = re.sub('[^가-힣]',' ', moon)
# print(moon)

# 2. 명사 추출 - konlpy.tag.hannanum() 의 nouns() 를 사용
import konlpy

hannanum = konlpy.tag.Hannanum()

a = hannanum.nouns('대한민국의 모든 영토는 한반도와 그 부속도서로 한다.')
# print(a)

nouns = hannanum.nouns(moon)
# print(nouns)

# 3. 데이터 프레임으로 전환
df_word = pd.DataFrame({'word' : nouns})
# print(df_word)

# 4. 단어 빈도표 생성
df_word['count'] = df_word['word'].str.len() # 단어의 길이변수 추가
# print(df_word)

# 3-1. 두 글자 이상 단어만 남기기
df_word = df_word.query('count >= 2')
# print(df_word.sort_values('count'))

# 4-2. 단어의 빈도 구하기
df_word = df_word.groupby('word', as_index=False) \
    .agg(n = ('word', 'count')) \
    .sort_values('n', ascending = False)
# print(df_word)

# 5. 단어 빈도 막대 그래프
top20 = df_word.head(20)
# print(top20)

plt.rcParams.update({'font.family' : 'Malgun Gothic',
                     'figure.dpi' : '120',
                     'figure.figsize' : [6.5, 6]})
# sns.barplot(data=top20, y='word', x='n', palette='Blues')
# plt.show()

### 워드 클라우드 : 단어 구름 모양 시각화
import wordcloud

# 1. 한글을 지원하는 폰트 설정
font = 'C:/Windows/Fonts/HMKMMAG.TTF' # 폰트 선정 후 경로 지정

# 2. 단어와 빈도를 담은 딕셔너리 생성
dic_word = df_word.set_index('word').to_dict()['n'] # 데이터 프레임을 딕셔너리로 변환
# print(dic_word)

# 3. 워드 클라우드 생성
from wordcloud import WordCloud
# wc = WordCloud(random_state=1234, # 난수 고정
#                font_path=font, # 폰트 설정
#                width=400, # 가로 크기
#                height=400, # 세로 크기
#                background_color='white') # 배경색
# img_wordcloud = wc.generate_from_frequencies(dic_word) # 워드 클라우드 생성
#
# plt.figure(figsize=(10,10)) # 가로 세로 배경 크기 설정
# plt.axis('off') # 테두리 선 없애기
# plt.imshow(img_wordcloud) # 출력물 지정
# plt.show() # 출력

### 워드 클라우드 모양 바꾸기
# 1. mask 생성
import PIL # 이미지 처리 패키지

icon = PIL.Image.open(path + 'cloud.png')

import numpy as np

img = PIL.Image.new('RGB', icon.size, (255,255,255))
img.paste(icon, icon)
img = np.array(img)
# print(img)

# 2. 워드 클라우드 생성
# wc = WordCloud(random_state=1234,
#                font_path=font,
#                width=400,
#                height=400,
#                background_color='white',
#                mask=img, # 마스킹 이미지 삽입
#                colormap='inferno') # 컬러맵 설정
# img_wordcloud = wc.generate_from_frequencies(dic_word) # 딕셔너리를 입력으로
#
# # 3. 출력
# plt.figure(figsize=(10,10))
# plt.axis('off') # 테두리 x
# plt.imshow(img_wordcloud)
# plt.show()

# -------------------------------------------------------------------------------------------------------------------
''' 웹 크롤링(Web Crawring) : 특정 사이트(웹)에서 원하는 정보를 긁어오는 행위 '''

'''
HTML : 웹 페이지의 표시를 위해서 개발된 지배적인 마크업 언어

- HTML 기본 구조

<!DOCTYPE html>
<html>
<head>
    <title> 웹 페이지의 제목 </title>
</head>
<body>
    <h1> 제목 </h1>
    <p> 들어갈 내용 <p>
</body>
</html>

-> tag : 열린 태그<>와 닫힌 태그 </> 사이에 컨텐츠를 위치해서 문서의 구조로 표현한 것
    1) h1 태그 : 문서의 제목 h1~h6
    2) p 태그 : 단락을 지정할 수 있는 태그
    3) img 태그 : 이미지를 표시할 수 있는 태그, 닫힌 태그가 필요 없음**
    4) input, button 태그 : 사용자의 입력이 필요할 때 input,
                            사용자가 클릭할 수 있는 버튼
    5) ul, ol, il 태그 : 리스트를 표현할 때 쓰는 태그
    6) div, span 태그 : 사용서 요소가 즉각적으로 나타나는 것과는 별개로 화면 내에서 아무런 역활은 없지만,
                        문서의 영역을 분리하고 인라인 요소를 감쌀 때 사용 
'''

# day 7 끝