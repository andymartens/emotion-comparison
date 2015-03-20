import flask
import urlparse
from matplotlib import pyplot as plt


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

    # do the graphs
    x = range(len(corpus1))
    y = [d * len(corpus2)-len(corpus1) for d in x]
    plt.figure()
    plt.plot(x,y)
    plt.savefig("static/corpus1.png")  #save graph to the static folder
    x = range(len(corpus1))
    y = [d / len(corpus2)-len(corpus1) for d in x]
    plt.figure()
    plt.plot(x,y)
    plt.savefig("static/corpus2.png")

    return flask.render_template("results.html",  #render the results.html page, which will get the graphs
                                 corpus_one = corpus1,  #will take the text fro the corpus if need it in results.html
                                 corpus_two = corpus2)



if __name__ == '__main__':  #this this just means run this whole file if call it from terminal?

    app.debug=True  #think this means that ea time i make change in py code, the server will reload automatically
    app.run(host='0.0.0.0')  #i think this makes the server publically avialable? but also makes 0.0.0.0 the web address i need to go to in order to view
