import csv
import requests
from bs4 import BeautifulSoup
import textstat
import re
from datetime import datetime


#list company profiles here (at least two):
company_list = ( "placeholer1.com",
                "placeholder2.com"

) #companies used for the study are not disclosed

# text cleaning function
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u0080-\uffff" 
                             "]+", re.UNICODE)


#open csv file, write headers
csv_file = open('datasettest.csv', 'w', newline='')

csv_writer = csv.writer(csv_file, delimiter =",",quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(["Company",
                     "Review Text",
                     #"Review Date",
                     "Rating",
                     "Useful",
                     "Verified",
                     "User Reviews",
                     "Wordcount Review",
                     #"Response Date",
                     "Readability",
                     "Response Text",
                     "Readability Response",
                     "Wordcount Response",
                     "Response Speed in Days"])


#get url part 1 
for profile in company_list:
    url1 = "https://www.trustpilot.com/review/{}".format(profile)
    r1 = requests.get(url1)
    soup1 = BeautifulSoup(r1.content, features="lxml")
    
#get page numbers + stop at last page + company name
    review_number_container = soup1.find_all("div",{"class":"reviews-overview card card--related"})[0].text
    m = re.search('English","reviewCount":"(.+?)"}',review_number_container)
    if m:
        review_number = m.group(1).replace(",","")
    page_number = int(review_number)//20

    company = soup1.find("span",{"class":"multi-size-header__big"}).text

#get urls & html text
    for i in range(2,int(page_number)):
        url = url1 + "?page={}".format(i)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, features="lxml")
        cards = soup.find_all("div",{"class":"review-card"})
        print(url)
   
#get data points: review text, date, rating, useful votes, response text & date, wordcounts & readability  
        for item in cards:
            if item.find_all("img")[0].get("alt")[0] == str('#'):
                pass #filter out reported reviews
            else:
                try:
                    review_title = item.find_all("h2",{"class":"review-content__title"})[0].text
                except:
                    review_title = ""
            
                try:
                    review_maintext = item.find_all("p",{"class":"review-content__text"})[0].text.replace("                ","")
                except:
                    review_maintext = "" 
                
                review_text = emoji_pattern.sub(r'', (review_title + ' ' + review_maintext)).replace("\n", " ").replace(",", "|").replace("   "," ").replace("  "," ")
       
                try:
                    review_date = item.find_all("div",{"class":"review-content-header__dates"})[0].text[20:30]
                except:
                    review_date = ""
                
                
                try:
                    verified_container = item.find_all("div",{"class":"review-content-header__review-verified"})[0].text
                    s = re.search('"isVerified":(.+?),',verified_container)
                    
                    if s.group(1) == str('true'):
                        verified = "1"
                    else:
                        verified = "0"
                        
                except:
                    verified = "0"
                
                try:
                    user_reviews = item.find_all("div",{"class":"consumer-information__review-count"})[0].text.replace("\n","")[0]
                except:
                    user_reviews = ""
               
                
                
                try:
                    rating = item.find_all("img")[0].get("alt")[0]    
                except:
                    rating = ""
                try:
                    useful = item.find_all("brand-find-useful-button")[0].get(":initial-find-useful-count")
                except:
                    useful = ""
       
                try:
                    response_text = emoji_pattern.sub(r'', (item.find_all("div",{"class":"brand-company-reply__content"})[0].text)).replace("\n", " ").replace("            ","").replace(",", "|").replace("君毅 駱","").replace("Δημήτρης Τσουκαλάς","")
                except:
                    response_text = ""
    
                try:
                    response_date = item.find_all("time-ago")[0].get("date")[:10]
                except:
                    response_date = ""
        
                wordcount_review = str(textstat.lexicon_count(review_text, removepunct=True))
        
                if str(textstat.lexicon_count(response_text, removepunct=True)) == str(0):
                    wordcount_response = ""
                else:
                    wordcount_response = str(textstat.lexicon_count(response_text, removepunct=True))
        
                readability = str(textstat.text_standard(review_text, float_output=True))
                
                if textstat.text_standard(response_text, float_output=True) == 0:
                    readability_response = ""
                else:
                    readability_response = str(textstat.text_standard(response_text, float_output=True))
                    
                
                
                try:
                    response_speed = str(datetime.strptime(str(response_date), '%Y-%m-%d') - datetime.strptime(str(review_date), '%Y-%m-%d'))[0]
                except:
                    response_speed = ""

                
                
                
# print in console     
                #print("review_text:" + review_text)
                #print("review_date:" + review_date)
                #print("rating:" + rating)
                #print("useful:" + useful)
                #print("response_text:" + response_text)
                #print("response_date:" + response_date)
                #print("wordcount_review:" + wordcount_review)
                #if wordcount_response == str(0):
                #    print("wordcount_response:"+ "")
                #else:
                #    print("wordcount_response:" + wordcount_response)
                #print("readability:" + readability)
                #print("response_speed:" + response_speed)

# write csv & close file      
                csv_writer.writerow([company,
                                     review_text,
                                     #review_date,
                                     rating,
                                     useful,
                                     verified,
                                     user_reviews,
                                     wordcount_review,
                                     readability,
                                     #response_date,
                                     response_text,
                                     readability_response,
                                     wordcount_response,
                                     response_speed])

csv_file.close()



#need to: 

#make separate file with averages PER COMPANY
    #avg rating
    #avg useful votes
    #avg wordcounts
    #avg readability
    #avg response speed in days