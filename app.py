from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)


@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        data = {key: value for key, value in request.form.items()}
        df = pd.DataFrame([data])
        scaler = joblib.load("./artifact/scaler.pkl")
        label_encoders = joblib.load("./artifact/label_encoders.pkl")
        linear = joblib.load("./model/linear_model.pkl")
        ridge = joblib.load("./model/ridge_model.pkl")
        mlp = joblib.load("./model/mlp_model.pkl")
        bagging = joblib.load("./model/bagging_model.pkl")

        for col in df.select_dtypes('object').columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
            
        # numerical_columns = df.select_dtypes(exclude=['object']).columns

        df = scaler.transform(df)

        linear_price = linear.predict(df)
        ridge_price = ridge.predict(df)
        mlp_price = mlp.predict(df)
        bagging_price = bagging.predict(df)

        # df_html = df.to_html()

        return render_template("result.html", linear_price = linear_price, ridge_price = ridge_price, mlp_price = mlp_price, bagging_price = bagging_price)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)