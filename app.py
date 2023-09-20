from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from output_parser import TagProblem_parser, TagProblem

nltk.download("stopwords")
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain





app = Flask(__name__)
classifier = pickle.load(open("models/classifier.pkl",'rb'))
tfidf = pickle.load(open("models/tfidf.pkl",'rb'))


def process_data(txt):

  # remove code snippets, HTML TAGS, Links
  clean_text = re.sub(r"<pre><code>.*?</code></pre>",r"CODE_SNIPPET",str(txt))
  clean_text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", clean_text)
  clean_text = re.sub(r"https?://\S+", "", clean_text)
  clean_text = re.sub(r"<.*?>", " ", clean_text)
  clean_text = " ".join(clean_text.split())

  # special character and punctuation
  clean_text = re.sub(r"[^A-Za-z0-9\s]+", "", clean_text)

  # reomoving numbers
  clean_text = re.sub(r"\b[0-9]+\b\s*", "", clean_text)

  # removing stop-words
  stopwords_ = set(stopwords.words("english"))
  tokens = clean_text.split()
  clean_tokens = [t for t in tokens if t.lower()=='c' or not t in stopwords_]
  clean_text = " ".join(clean_tokens)

  # SnowBall Stemming
  snowball = SnowballStemmer(language='english')
  text=''
  for word in clean_text.split():
    text=text+snowball.stem(word)+' '

  return text

def gpt_opt(information) -> TagProblem:
    summary_template = """
    
    here is a tag_list : ['javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', 'ios']
    
    Classify the user entered problem statement {information} into the tags from the tag_list.
    
    show the tags into which problem statement is classified.
    
    If information does not belong to any of the tags into tag_list then print "No Tags given"
    \n{format_instructions}    
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template, partial_variables={"format_instructions":TagProblem_parser.get_format_instructions()})
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(information=information)
    return TagProblem_parser.parse(result)


def print_tags(text1,text2):
    tag_list= ['javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', 'ios']
    ext_test = [{"Title": text1, "Body": text2}]
    ext_test = pd.DataFrame.from_dict(ext_test)
    ext_test["Title_Body"] = ext_test["Title"].str.cat(ext_test["Body"], sep=" ")
    ext_test['Title_Body'] = ext_test['Title_Body'].apply(process_data)
    ext_test = tfidf.transform(ext_test['Title_Body'])
    pred = classifier.predict(ext_test).toarray()
    pred_tags = [tag_list[i] for i in range(10) if int(pred[0][i])==1]
    pred_tags = " ".join(pred_tags)

    return pred_tags




@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    title = request.form.get('title-content')
    body = request.form.get('body-content')
    title_body = title + " " + body
    pred = print_tags(title, body)
    gpt = gpt_opt(title_body)

    return render_template("index.html", title=title, body=body, pred=pred, gpt=gpt.tags)



if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)