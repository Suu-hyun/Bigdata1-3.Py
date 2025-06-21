""" 25 - 06 - 21 """
import time

import requests

web = requests.get("https://www.daangn.com/kr/buy-sell/?in=%EB%B6%80%EC%95%94%EB%8F%99-6")
# print(web.text) # 해당 웹페이지 텍스트만 출력

''' BeautifulSoup 라이브러리 : HTML 문서에서 원하는 부분만 추출하게 해주는 기능 라이브러리

from bs4 import BeautifulSoup(BeautifulSoup는 클레스) <- 사용법
'''
from bs4 import BeautifulSoup

soup = BeautifulSoup(web.content, 'html.parser')
'''
파싱(parser) : HTML 또는 XML 문서를 구문 분석하여 파이썬 객체로 변환해주는 작업
'''
# print(soup.h1)

### 필요한 텍스트만 추출

# 1. ul 태그의 하위 항목을 모두 뽑아오고 싶을 때
# for child in soup.ul.children: # for 반복문의 마무리는 항상 콜론(:)으로 해준다.
#     print(child)
#     time.sleep(2)

# 2. find_all() : 지정한 태그의 모든 값을 가져오는 함수
# print(soup.find_all('div'))
# for f in soup.find_all('div'):
#     print(f)
#     time.sleep(3)

# for menu in soup.find_all('li'):
#     print(menu.get_text())

# 3. 정규식을 활용하는 방법 - <ol> 든 <ul> 다 포함된 리스트를 긁어오고 싶을 때
import re
# re.compile("") 정규식 객체를 리턴해주는 함수(공부필요)
# for f in soup.find_all(re.compile('[ou]l')): # ol 태그 ul 태그
#     print(f)

# 4. 리스트 활용 - 원하는 태그를 직접 지정해서 뽑는 경우  h1, a 태크만 보고 싶을 때
# for f in soup.find_all(['h1','a']):
#     print(f)

# 5. HTML 속성 활용 - 속성(class, id....)을 지정해서 뽑고 싶을 때 <- 추천 방법
# a = soup.find_all(attrs={'data-gtm' : 'search_article'})
# for i in a :
#     print(i.get_text())

# 6. CSS 선택자를 통해 원하는 부분 가져오고 싶을 때
''' CSS : 웹 페이지의 레이아웃 스타일을 디자인하고 정의하는데 사용하는 스타일 시트 언어, HTML 과 함께 사용 '''
# a = soup.select('a:nth-child(1)')
# main-content > div.wv2v230.sprinkles_paddingLeft_4_base__1byufe8g6.sprinkles_paddingLeft_10_small__1byufe8gr.sprinkles_paddingLeft_16_medium__1byufe8h4.sprinkles_paddingLeft_20_large__1byufe8h9.sprinkles_paddingRight_4_base__1byufe8e2.sprinkles_paddingRight_10_small__1byufe8en.sprinkles_paddingRight_16_medium__1byufe8f0.sprinkles_paddingRight_20_large__1byufe8f5.sprinkles_backgroundColor_layerDefault__1byufe8n.sprinkles_width_full_base__1byufe84q > div > section > div > div.b4c4lz3.sprinkles_width_full_base__1byufe84q.b4c4lz5 > a:nth-child(43) > div > div.sprinkles_display_flex_base__1byufe82i.sprinkles_gap_1_base__1byufe8qe.sprinkles_alignItems_flexStart_base__1byufe8se.sprinkles_flexDirection_column_base__1byufe8te > div.sprinkles_display_flex_base__1byufe82i.sprinkles_color_neutral__1byufe81.sprinkles_width_full_base__1byufe84q.sprinkles_alignItems_flexStart_base__1byufe8se.sprinkles_flexDirection_column_base__1byufe8te.sprinkles_gap_0\.5_base__1byufe8s2 > span.sprinkles_fontSize_200_base__1byufe8uu.sprinkles_fontWeight_regular__1byufe81x.sprinkles_lineHeight_body\.medium_base__1byufe8w6.sprinkles_color_neutral__1byufe81.lm809sh.sprinkles_overflow_hidden__1byufe819
# print(a)

# 7. 텍스트만 읽어오고 싶을 때
# for x in range(0,10):
#     print('현재 x:', x)
#     print(soup.select('li')[x].get_text())
#     time.sleep(2)

# ------------------------------------------------------------------------------------------------------------------------
''' 날씨 요약 프로그래밍 - 웹 크롤링 '''

## import
# import datetime
# from bs4 import BeautifulSoup
# import urllib.request
# import requests
#
# ## 현재 시간을 출력하고 본인 스타일에 맞게 출력문 수정
# now = datetime.datetime.now() # 현재 시각
# # print(now)
# nowDate = now.strftime('%Y년 %m월 %d일 %H시 %M분 입니다.') # 현재 시간 문자열 포맷 변경
# print(nowDate)
# print('■'*100)
# print('\t\t\t\t\t\t\t\t ※ Python Web Crawling Project ※') # \t : 들여쓰기 1칸 = 스페이스바 4번
# print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t ID : SSHadmin')
# print('■'*100)
# print('반갑습니다. 현재 시간은', nowDate, '\n') # \n :  줄 바꿈
# print("\t Let Me Summarize Today's Info \n")

## 서울 날씨
# print('#오늘의 #날씨 #요약 \n')
# webpage = urllib.request.urlopen('https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%EC%84%9C%EC%9A%B8%EB%82%A0%EC%94%A8&ackey=d6sf9znk')
# soup = BeautifulSoup(webpage,'html.parser')
# temps = soup.find('strong', '') # 온도
# # print(temps)
# cast = soup.find('p', 'summary')
# # print(cast)
# print('--> 서울 날씨 :', temps.get_text(),cast.get_text())
#
# ## 부산 날씨
# webpage_busan = urllib.request.urlopen('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EB%B6%80%EC%82%B0%EB%82%A0%EC%94%A8&oquery=%EC%84%9C%EC%9A%B8%EB%82%A0%EC%94%A8&tqi=jais2wqosTCssRtJiqNssssstYs-501578&ackey=qugnyuhn')
# soup_b = BeautifulSoup(webpage_busan, 'html.parser')
# temps_b = soup_b.find('strong', '')
# cast_b = soup_b.find('p', 'summary')
# print('--> 부산 날씨 :',temps_b.get_text(),cast_b.get_text())
#
# ## 제주도 날씨
# webpage_jejudo = urllib.request.urlopen('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EC%A0%9C%EC%A3%BC%EB%8F%84%EB%82%A0%EC%94%A8&oquery=%EB%B6%80%EC%82%B0%EB%82%A0%EC%94%A8&tqi=jaiuldqo1e8ssDFdMJVssssssXK-472064&ackey=7v7son5r')
# soup_j = BeautifulSoup(webpage_jejudo, 'html.parser')
# temps_j = soup_j.find('strong', '')
# cast_j = soup_j.find('p', 'summary')
# print('--> 제주도 날씨 :',temps_j.get_text(),cast_j.get_text())


# 이 뒤의 수업은 코렙을 이용함(2시36분)
# DAY 8 끝




