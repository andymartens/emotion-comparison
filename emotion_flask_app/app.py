import flask
import urlparse
from matplotlib import pyplot as plt
import pickle
import emotion_extraction as ee


app = flask.Flask(__name__)  #app is the web app

#if someone goes to the root/base address (which ends in a /),
#it'll send a GET request and this will run this function that renders the input.html file
@app.route("/")  # ("/" is saying what url should trigger the function below.)
def homepage():
    return flask.render_template("input.html")  #looks in the templates folder for input.html


#when a POST request comes in from url /analyze, run function below:
@app.route("/analyze", methods=["GET", "POST"])  #think allows this to receive a POST request, i.e., get data
def analysis():
    data = flask.request.get_data()  #the form on input.html is post in the data to flask
    parsed_data = urlparse.parse_qs(data)
    corpus1 = parsed_data['corpus1'][0]
    corpus2 = parsed_data['corpus2'][0]
    corpus1 = corpus1.decode('utf-8')
    corpus2 = corpus2.decode('utf-8')


    # len of list was 1. couldn't create list of text docs.
    # corpus1_list = []
    # corpus2_list = []
    # corpus1_list.append(corpus1)
    # corpus2_list.append(corpus2)

    # but len of corpus1 as it is not is 230,000
    # how to divide up into documents? how to enter into textbox on web app?
    # split them on """ and then join back into a list? first need to put the
    # corpus in a list, i.e., corpus_list[corpus1], and then split on triple quotes
    corpus1_list = corpus1.split('""", """')
    corpus2_list = corpus2.split('""", """')

    # didn't work, couldn't process:
    # corpus1_list = list(corpus1)
    # corpus2_list = list(corpus2)


    # Retrieve emotion-words to variations dict:
    with open('root_to_variations_dict.pkl', 'r') as picklefile:
        root_to_variations_dict = pickle.load(picklefile)
    # Retrieve emotion-words to ratings dict:
    with open('corresponding_root_to_ratings_dict.pkl', 'r') as picklefile:
        corresponding_root_to_ratings_dict = pickle.load(picklefile)


    #takes data from two corpora and saves two graphs to static folder
    #later: allow user to input name of each corpus and replace 'Corpus 1' and 'Corpus 2' w those names
    ee.corpuses_to_plot(corpus1_list, corpus2_list, 'Corpus 1', 'Corpus 2', root_to_variations_dict, corresponding_root_to_ratings_dict)

    return flask.render_template("results.html",  #render the results.html page, which will get the graphs
                                 corpus_one = len(corpus1_list),  #will take the text fro the corpus if need it in results.html
                                 corpus_two = len(corpus2_list))


if __name__ == '__main__':  #this this just means run this whole file if call it from terminal?

    app.debug=True  #think this means that ea time i make change in py code, the server will reload automatically
    app.run(host='0.0.0.0')  #i think this makes the server publically avialable? but also makes 0.0.0.0 the web address i need to go to in order to view
