from flask import Flask, jsonify, render_template,request,flash,redirect,url_for,session
import sqlite3
import subprocess
from Anupa.Chatbot.Chatbot_001 import *
from main import run_detection

app = Flask(__name__)
app.secret_key="123"

con=sqlite3.connect("database.db")
con.execute("create table if not exists user(pid integer primary key,name text,password text,rdusername text,inusername text,inpassword text,spusername text)")
con.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method == 'POST':
        name = request.form['name']
        entered_password = request.form['password']
        con = sqlite3.connect("database.db")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM user WHERE name=?", (name,))
        user_data = cur.fetchone()

        if user_data:
            stored_password_with_salt = user_data['password']
            stored_salt = stored_password_with_salt[:10]  # Extract the first 10 characters as the stored salt
            entered_password_with_salt = stored_salt + entered_password

            if stored_password_with_salt == entered_password_with_salt:
                if user_data:
                    session["name"]=user_data["name"]
                    session["rdusername"]=user_data["rdusername"]
                    session["inusername"]=user_data["inusername"]
                    session["inpassword"]=user_data["inpassword"]
                    session["spusername"]=user_data["spusername"]
                    # session["subscription"]=user_data["subscription"]
                    # return redirect("detect")

                    # Determine the route to redirect based on the user's subscription
                    if user_data["subscription"] == "pro":
                        return redirect("detect")
                    else:
                        return redirect("detect_standard")
                
            else:
                # Incorrect password
                return "Incorrect password"
        else:
            # User not found
            return "User not found"

        # con.close()
    
    return redirect(url_for("index"))

#pro version home
@app.route('/detect',methods=["GET","POST"])
def detect():
    return render_template("detect.html")

#standard version home
@app.route('/detect_standard',methods=["GET","POST"])
def detect_standard():
    return render_template("detect_standard.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            name=request.form['name']
            password=request.form['password']
            rdusername=request.form['rdusername']
            inusername=request.form['inusername']
            inpassword=request.form['inpassword']
            spusername=request.form['spusername']
            # subscription=request.form['subscription']

            con=sqlite3.connect("database.db")
            cur=con.cursor()
            cur.execute("insert into user(name,password,rdusername,inusername,inpassword,spusername)values(?,?,?,?,?,?)",(name,password,rdusername,inusername,inpassword,spusername))
            con.commit()
            flash("Record Added  Successfully","success")
        except:
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("index"))
            con.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    # return redirect(url_for("landingpg"))
    return redirect(url_for("index"))

@app.route('/base', methods=['POST'])
def perform_detection():
    # Receive data from the request payload
    data = request.get_json()

    # Extract the relevant data from the received JSON
    name = data.get('name')
    rdusername = data.get('rdusername')
    inusername = data.get('inusername')
    spusername = data.get('spusername')
    inpassword = data.get('inpassword')

    # Create a list of arguments to pass to main.py
    main_script_args = [
        "main.py",
        "--name", name,
        "--rdusername", rdusername,
        "--inusername", inusername,
        "--spusername", spusername,
        "--inpassword", inpassword
    ]
    try:
        x=run_detection(name, rdusername, inusername, spusername, inpassword)
        print("run detection success")
    finally:
        print("now re direct please")
        return redirect(url_for("index"))
    # Run main.py using subprocess and pass the arguments
    # subprocess.run(["python"] + main_script_args)

    return render_template('base.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid

    response = all_func(text,False)# (see comment in line 3)
    # response = "hii"
    print(response)
    message ={"answer": response}
    return jsonify(message)

@app.route('/payment_pro')
def payment_pro_v():
    return render_template('payment_pro_v.html')


if __name__ == '__main__':
    app.run(debug=True)