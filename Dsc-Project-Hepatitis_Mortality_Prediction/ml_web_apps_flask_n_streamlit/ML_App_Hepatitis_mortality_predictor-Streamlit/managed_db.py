# DB
import sqlite3
conn = sqlite3.connect('usersdata.db')
c = conn.cursor()

# Functions

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data



def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data