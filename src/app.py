import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request
from model import recommend as get_recommendations

_base = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_base, "..", "frontend", "pages"),
    static_folder=os.path.join(_base, "..", "frontend", "static"),
)


@app.route("/")
def home():
    return render_template("mainpage.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("mainpage.html")

    results = get_recommendations(query)
    if isinstance(results, str):
        return render_template("mainpage.html", error=results)

    return render_template("result.html", movies=results, query=query)


if __name__ == "__main__":
    app.run(debug=True)
