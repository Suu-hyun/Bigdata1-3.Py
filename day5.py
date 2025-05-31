''' 25 - 03 - 31 강의 자료 '''
import pandas as pd
import numpy as np
from pandas.io.common import file_path_to_url
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

'''
변수(Variable)
'''

### 여러 값으로 구성된 변수 만들기
var1 = [1,2,3] # list(리스트) : 대괄호 안에 여러 가지 값을 입력해서 생성하는 형태
var2 = [4,5,6]

# print(var1)
# print(var2)
#
# print(var1 + var2)
'''
리스트의 연산은 리스트를 합쳐(연결시켜)준다.

주석 단축기 : Ctrl + ?
'''

### 문자 변수 생성
str1 = 'a'
# print(str1)

str2 = 'text'
# print(str2)

str2_2 = '13' # 따옴표 안에 들어간 숫자는 문자로써 기능을 한다.
# print(str2_2)

str3 = 'Hello World !'
str4 = ['a', 'b', 'c']
str5 = ['Hello!', 'World', 'is', 'good']

# 문자 변수 결합
# print(str2 + str3) # 문자열 더하기 연산은 문자열을 연결시켜준다.
str6 = '100'
# print(str2_2 + str6)

# -----------------------------------------------------------------------------------------------------
''' 함수(function) : 입력을 받아서 출력을 내주는 수

- 값을 넣으면 특정한 기능을 수행해서 처음과 다른 값을 만드는 수
- 괄호 안에 입력을 받는다.

print()
'''

### 함수 이용하기
x = [1, 2, 3]
# print(sum(x)) # sum() : 합계 함수
# print(max(x)) # max() : 최댓값

## 함수의 결과물로 새 변수 만들어서 활용한다. ***
x_sum = sum(x)
# print(x_sum)

# ---------------------------------------------------------------------------------------------------------------------
'''
패키지(Package)

- 변수나  함수가 여러 개 들어 있는 꾸러미
- 패키지 안의 함수를 사용하려면 패키지 설치 먼저
- 아나콘다에 주요 패키지 대부분 들어 있음

            패키지 설치 -> 패키지 로드 -> 패키지 함수 사용
'''

import seaborn # 시각화 패키지
import matplotlib.pyplot as plt # 패키지 명이 길명 as 구문으로 줄임말을 사용해서 쓸 수 있다.
var = ['a', 'a', 'b', 'c', 'd', 'd', 'd']

# seaborn.countplot(x=var) # . 은 접속 연산자
# plt.show() # 그래프 출력 함수

## seaborn 패키지의 titanic 데이터로 그래프 생성
''' 여러 패키지들은 패키지 안의 함수들을 사용해볼 수 있게 예제 데이터를 포함시킨다. '''
df = seaborn.load_dataset('titanic')
# print(df)

# seaborn.countplot(data=df, x='sex')
# plt.show()

# x축에 class 선실 등급 넣어보기
import seaborn as sns
# sns.countplot(data=df, x='class')
# plt.show()

# x축에 class, alive 별 색 표현
# sns.countplot(data=df, x='class', hue='alive')
# plt.show()

# ---------------------------------------------------------------------------------------------------------------
'''
모듈(Module) : 파이썬 파일 (.py)
'''

# import sklearn.metrics
# sklearn.metrics.accuracy_score()

# '모듈명.함수명()' 으로 함수 사용 -> from 패키지명 import 모듈, 함수명

# from sklearn import metrics
# metrics.accuracy_score()

# '함수명()' 으로 함수 사용 -> 정확히 사용할 함수 하나 또는 몇 개만 쓸 때

# from sklearn.metrics import accuracy_score, dcg_score
# accuracy_score()
# dcg_score()

### 패키지 설치 - 아나콘다 환경
'''
1. 파이참 왼쪽 아래 버튼 중 Terminal 접속
2. local 옆에 있는 아래 화살표 New Session 버튼 눌러서 Command Prompt 버튼 클릭
3. () 안에 있는 base 환경 -> 아나콘다 환경 설정
4. 내가 사용하고 있는 환경에서 패키지 설치
5. 파이썬에서 기본적으로 패키지 설치 함수 pip install 패키지명
'''

# import pydataset
# a = pydataset.data()
# print(a)

# --------------------------------------------------------------------------------------------------------------
'''
데이터 프레임(Data Frame)

- 데이터를 다룰 때 가장 많이 사용하는 데이터 형태
- 행과 열로 구성된 사각형 모양의 표처럼 생김
- 엑셀과 비슷한 구조 (vs 열의 이름(변수명)이 포함된 형태)

- 열은 속성이다 -> 컬럼(column) 또는 변수(Variable) 이라 불림
- 행은 한 사람의 정보다 -> 로(row) 또는 케이스(case) 라고 불림
'''

### 데이터 입력해서 데이터 프레임 만들기
import pandas as pd

df = pd.DataFrame({'name' : ['김지훈', '이유진', '빅동현', '김민지'],
                   'english' : [90,80,60,70],
                   'math' : [50,60,100,20]
                   })
# print(df)
''' 파이썬은 숫자를 처음 셀 때 0부터 센다. '''

# 특정 변수의 값을 추출하기 - 필요한 것만 뽑아내기
# print(df['english'])

# 변수의 값으로 합계를 구하기
a = sum(df['english'])
# print(a)

# 변수의 값으로 평균 구하기
b = sum(df['math']) / 4
# print(b)

### 외부 데이터 활용/로드
''' 경로 변수 : 작업을 시작하기 전에 데이터가 존재하는 경로를 미리 변수로 지정해놓는 기법 '''
path = 'C:/PData/'

df_exam = pd.read_excel(path + 'excel_exam.xlsx')
# print(df_exam)

# 영어 점수 평균
a = sum(df_exam['english']) / 20
# print('학생들의 영어 평균 :', a)

# len() 함수를 사용해서 평균 구하기 자동화
x = [1,2,3,4,5,6,6,7,8,34,3,23,43,23,2,431,21,3,123]
# print(len(x)) # 해당 값의 길이를 출력

b = sum(df_exam['math']) / len(df_exam)
# print(b)

### 엑셀 파일의 첫 번째 행이 변수명이 아닌 경우
df_exam_no = pd.read_excel(path + 'excel_exam_novar.xlsx', header=None) # head=None을 쓰면 변수명을 0부터 지정해서 나옴
# print(df_exam_no)

### 엑셀 파일에 시트가 여러 개 있는 경우
# df_exam = pd.read_excel('excel_exam.xlsx', sheet_name='sheet2')
# df_exam = pd.read_excel('excel_exam.xlsx', sheet_name=3) # 3 : 네번째 시트

### csv 파일 불러오기
csv_exam = pd.read_csv(path + 'exam.csv')
# print(csv_exam)

# -------------------------------------------------------------------------------------------------------------------
''' 분석 기초 - 데이터 파악하기 '''
'''
1. head() : 데이터의 앞 부분 출력
2. tail() : 데이터의 뒷 부분 출력
3. shape : 행, 열 개수 출력
4. info() : 변수 속성 출력
5. describe() : 요약 통계량 출력
'''

### 데이터 파악 준비 - 패키지, 데이터 로드
import pandas as pd

exam = pd.read_csv(path + 'exam.csv')

### head() - 앞 부분 출력
a = exam.head() # 앞 부분 5개 행만 출력
# print(a)

#  앞에서부터 15행까지 출력하기
b = exam.head(15)
# print(b)

### tail() - 뒷 부분 출력
a = exam.tail(7) # 뒤에서부터 7개 행 출력
# print(a)


### shape : 데이터가 몇 행, 몇 열로 구성되어 있는지 알아보기
# print(exam.shape)

### info() - 변수 속성 파악
# print(exam.info())
'''
Non-Null Count : 결측치(누락된 값)를 제외하고 구한 값의 개수
변수 속성 : int64(정수), float64(실수), object(문자), datetime64(날짜 시간)

64 : 64 비트
 - 1비트로 두 개의 값 표현 가능
 - int64 : 2 ^ 64 개의 정수 표현이 가능
'''

### describe() - 요약통계량
# print(exam.describe())
'''
std : 표준편차 : 변수의 값들이 평균에서 떨어진 정도를 나타낸 값
count : 빈도 : 값의 개수
'''

### mpg 데이터 파악
mpg = pd.read_csv(path + 'mpg.csv')
# print(mpg)

# print(mpg.info)
# print(mpg.describe()) # 문자 변수와 수치형 변수가 섞여있을 때 수치형 변수만 통계량을 내준다.
# print(mpg.describe(include = 'all'))
'''
unique : 고유값 빈도 - 중복을 제거한 범주의 개수
top : 최빈값 - 개수가 가장 많은 값
freq : 최빈값 빈도 - 개수가 가장 많은 값의 개수
'''

'''
            sum()           pd.read_csv()               df.head()
            내장함수            패키지 함수                  메서드

1. 내장함수 : 파이썬 자체에 내장되어 있는 함수
    - sum(), max() ...
2. 패키지 함수 : 패키지를 설치하고 로드해야 쓸 수 있는 함수
    -plt.show(), pd.read_csv() ...
3. 메서드(Method) - 변수 자체가 지니고 있는 함수
    - df.head(), df.info()
    - 변수의 자료 구조에 따라 사용 가능한 메서드가 다르다.
4. 어트리뷰트(attribute) : 변수가 지니고 있는 값
    - df.shape
'''

### 변수명 수정
df_raw = pd.DataFrame({'var1' : [1,2,1],
                       'var2' : [2,3,2]})
# print(df_raw)

# *** 데이터 프레임 복사본 만들기
'''
- 오류가 발생하더라도 원 상태로 되돌릴 수 있다.
- 데이터를 비교하면서 변형되는 과정을 검토할 수 있다.
'''
df_new = df_raw.copy() # 복사본 만들기
# print(df_new)

# 변수명 수정
df_new = df_new.rename(columns={'var2' : 'v2'})
# print(df_new)

### 파생 변수(derived variable) : 기존의 변수를 변형시켜서 만드는 새로운 변수

# mpg 데이터에서 통합 연비 변수 생성
# print(mpg.head(20))

mpg['total'] = (mpg['cty'] + mpg['hwy']) / 2
# print(mpg.head())

# 파생변수 분석
a = sum(mpg['total']) / len(mpg)
# print('미국 자동차들의 전체 통합 연비 평균 :', a)
b = mpg['total'].mean()
# print(b)

### 조건문을 활용해서 파생변수 생성
# 1. 기준값 정하기
a = mpg['total'].describe()
# print(a) 평균인 20.14를 기준으로 채택

# 2. 합격 판정 변수 만들기
import numpy as np

mpg['test'] = np.where(mpg['total'] >= 20, '합격', '불합격')
# print(mpg.head(30))

# 3. 빈도표로 합격 판정 자동차 수 살펴보기
a = mpg['test'].value_counts() # 빈도표 만드는 함수
# print(a)

# 4. 막대그래프로 표현
c_test = mpg['test'].value_counts()
c_test.plot.bar() # 연비 합격 빈도 막대 그래프 생성
# plt.show()
''' 파이썬에서 그래프로 시각화를 할 때 한글이 깨지는 경우 ?

-> 한글을 지원하는 폰트를 지정
plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 한글 폰트 지정
위의 코드를 코드 맨 위로 올려준다.
'''

# 5. 축 이름 회전하기
# c_test.plot.bar(rot=0) # 축 이름 수평으로 만들기
# plt.show()


### 중첩 조건문 활용
'''
A 등급 : 30 이상
B 등급 : 20~29
C 등급 : 20 미만
'''

# 연비 등급 변수 생성
mpg['grade'] = np.where(mpg['total'] >= 30, 'A',
                        np.where(mpg['total'] >= 20, 'B', 'C'))
# print(mpg.head())

# 빈도표와 막대 그래프로 연비 등급 확인
count_grade = mpg['grade'].value_counts() # 등급 빈도표 생성
# print(count_grade)

# count_grade.plot.bar(rot=0)
# plt.show()

# 등급 빈도표 알파벳 순으로 정렬
count_grade = mpg['grade'].value_counts().sort_index()
# 메서드 체이닝 : . 을 사용해서 메서드를 계속 이어가는 방법
# print(count_grade)

# count_grade .plot.bar(rot=0)
# plt.show()

### 목록에 해당하는 행으로 변수 만들기
# 버티컬 바(|) : 또는 을 의미하는 기호

# compact 차, suvcompact, 2seater 차는 소형이고 나머지는 큰 차로 분류하기
mpg['size'] = np.where((mpg['category'] == 'compact') |
                       (mpg['category'] == 'subcompact') |
                       (mpg['category'] == '2seater'),
                       'SMALL', "LARGE")
a = mpg['size'].value_counts()
# print(a)
''' np.where() 에 여러 조건 입력할 때 각 조건에 괄호 입력을 해줘야한다. '''

# 목록에 해당하는 행으로 변수 생성 - df.isin()
mpg['size'] = np.where(mpg['category'].isin(['compact',
                                             'subcompact',
                                             '2seater']), 'small','large')
a = mpg['size'].value_counts()
# print(a)

# -----------------------------------------------------------------------------------------------------------
''' 데이터 가공/전처리 - 분석하기 쉽게 데이터를 원하는 형태로 가공해주는 작업 '''

''' 전처리 함수(데이터 가공 함수들)
pandas : 전처리 작업에 가장 많이 사용되는 패키지

1. query() : 행 추출
2. 데이터명[] : 열 추출
3. sort_values() : 정렬
4. groupby() : 집단 별로 나누기
5. assign() : 파생 변수 추가
6. agg() : 통계치 산출
7. merge() : 데이터 합치기(열 기준) 
8. concat() : 데이터 합치기(행 기준)
'''

### 조건에 맞는 데이터만 추출하기 - df.query()
import pandas as pd
exam = pd.read_csv(path + 'exam.csv')
# print(exam)

# exam 데이터에서 nclass 가 1인 경우만 추출
a = exam.query('nclass == 1')
# print(a)

# 1반이 아닌 경우
a = exam.query('nclass != 1') # 파이썬에서 !는 부정을 의미
# print(a)

## 초과, 미만, 이상, 이하 조건 걸기
# 1. 수학 점수가 50점을 초과한 경우
a = exam.query('math > 50')
# print(a)

## 여러 조건을 충족하는 행 추출
# 1. 1반이면서 수학 점수가 50점을 넘긴 학생
a = exam.query('nclass == 1 & math >= 50')
# print(a)

# 수학 점수가 80점 이상이거나 영어 점수가 80점 이상인 경우
b = exam.query('math >= 80 | english >= 80')
# print(b)

## 목록에 해당하는 행 추출
# 1, 4, 5 반에 해당하면 추출
a = exam.query('nclass == 1 | nclass == 4 | nclass == 5')
# print(b)
b = exam.query('nclass in [1,4,5]')
# print(b)

## 문자 변수를 사용해서 조건에 맞는 행 추출
''' 전체 조건과 추출할 문자에 서로 다른 문양의 따옴표를 입력하면 된다. '''
df = pd.DataFrame({'sex' : ['F', 'M', 'F', 'M'],
                   'country' : ['Korea', 'China', 'Japan', 'USA']})
# print(df)

a = df.query('sex == "F" & country == "Korea"')
print(a)
b = df.query("sex == 'M' & country == 'China'")
# print(b)

''' 파이썬에서 사용하는 기호
- 논리 연산자
1. < : 작다
2. <= : 작거나 같다
3. == : 같다
4. != : 같지 않다
5. | : 또는
6. & : 그리고
7. in : 포함 연산자, 매칭 확인

- 산술 연산자
1. +, -
2. / : 나누기
3. // : 나눗셈의 몫
4. % : 나눗셈의 나머지
5. * : 곱하기
6. ** : 제곱
'''

# day 5 끝 오류 많이 발생함(오타 체크 필요)