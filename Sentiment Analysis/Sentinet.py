import csv
import re
import nltk
file=open("senti.csv","r")
reader = csv.DictReader(file, delimiter=',')
scores=dict()
final_scores = dict();
for i,row in enumerate(reader):
    score= float(row["PosScore"])-float(row["NegScore"])
    terms=row["SynsetTerms"].split(" ")
    for term in terms:
        check=term.split("#")
        key=check[0]+"#"+row["POS"]
        if key in final_scores:
         final_scores[key].append({check[1]:score})
        else:
            final_scores[key]=list()
            final_scores[key].append({check[1]:score})
for k,v in final_scores.items():
    score,total=0.0,0.0
    for term in v:
        for k1,v1 in term.items():
            score += float(v1)/float(k1)
            total += 1.0 / float(k1);
    score /= total;   
    scores[k]=score
    
#for i,(k,v) in enumerate(scores.items()):
#    if v>0:
#     print(str(k)+" "+str(v))

for i,(k,v) in enumerate(final_scores.items()):
     if i<10:
        print(str(k)+" "+str(v))

def ch_pos(pos):
 lis3=[]
 for word in pos:
  if re.match('nn+',word[1].lower()): lis3.append(word[0]+'#n')
  if re.match('vb+',word[1].lower()): lis3.append(word[0]+'#v')
  if re.match('jj+',word[1].lower()): lis3.append(word[0]+'#a')
  if re.match('rb+',word[1].lower()): lis3.append(word[0]+'#a')  
 return lis3
#print(pos2)    
        
neg_lis=list()

with open(r"reviews.txt","r") as neg:
 for line in neg:
  neg_lis.append(line.split("|"))

final_score=[]

for line in neg_lis:
 tot_score=0
 tokens=[word for word in nltk.word_tokenize(line[0])if word not in nltk.corpus.stopwords.words()]
 pos=nltk.pos_tag(tokens)
 pos2=ch_pos(pos)
 for item in pos2: 
    if item in scores:
     tot_score+=float(scores[item])   
 final_score.append(tot_score)
print("Accuracy is:")
print(len([score for score in final_score if score<0])/390)
