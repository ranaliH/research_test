# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:15:47 2023

@author: Imtiaz
"""
from flask import Flask, render_template,request,flash,redirect,url_for,session
import sqlite3

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
    if request.method=='POST':
        name=request.form['name']
        password=request.form['password']
        con=sqlite3.connect("database.db")
        con.row_factory=sqlite3.Row
        cur=con.cursor()
        cur.execute("select * from user where name=? and password=?",(name,password))
        data=cur.fetchone()

        if data:
            session["name"]=data["name"]
            session["rdusername"]=data["rdusername"]
            session["inusername"]=data["inusername"]
            session["inpassword"]=data["inpassword"]
            session["spusername"]=data["spusername"]
            return redirect("detect")
        else:
            flash("Username and Password Mismatch","danger")
    return redirect(url_for("index"))


@app.route('/detect',methods=["GET","POST"])
def detect():
    return render_template("detect.html")

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
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)