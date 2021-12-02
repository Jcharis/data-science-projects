import streamlit as st
import joblib
import os
import datetime
from PIL import Image
import altair as alt
import pandas as pd
import numpy as np
import datetime

# Storage in SQL
from sqlmodel import Field,Session,SQLModel,create_engine,select
from typing import Optional

class PredictionStore(SQLModel,table=True):
	__table_args__ = {'extend_existing':True}
	id: Optional[int] = Field(default=None,primary_key=True)
	text_issue: str
	prediction: str 
	probability: float
	date: Optional[datetime.datetime] = None

engine = create_engine("sqlite:///data/data.db") # Path to DB
SQLModel.metadata.create_all(engine) # Create the DB and Table



# Fxn To Load Model
def load_model(model_file):
	model = joblib.load(open(os.path.join(model_file),"rb"))
	return model

ISSUE_CLASSIFIER = load_model("models/pipe_dt_cv_gh_issue_classifier_27_nov_2021.pkl")

def plot_prediction_proba(term):
	pred_proba_df = pd.DataFrame({'Probabilities':ISSUE_CLASSIFIER.predict_proba([term])[0],'Classes':ISSUE_CLASSIFIER.classes_})
	c = alt.Chart(pred_proba_df).mark_bar().encode(
	    x='Classes',
	    y='Probabilities',
	    color='Classes'
	)
	st.altair_chart(c)

def load_image(image_file):
	img = Image.open(image_file)
	st.image(img)



def main():
	st.title("Streamlit App")
	st.subheader("Hello Streamlit")

	menu = ["Home","Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		# Form
		with st.form(key='myForm'):
			text_issue = st.text_area("Enter Github Issue Here")
			submit_button = st.form_submit_button(label='predict')


		if submit_button:
			# Layout
			col1,col2 = st.columns(2)
			prediction = ISSUE_CLASSIFIER.predict([text_issue])
			pred_proba = ISSUE_CLASSIFIER.predict_proba([text_issue])
			probabilities = dict(zip(ISSUE_CLASSIFIER.classes_,np.round(pred_proba[0],3)))


			with col1:
				st.info("Original Issue")
				st.text(text_issue)

				st.info("Prediction")
				st.write(prediction[0])
				st.write(probabilities)
				# Store Results
				pred_data = PredictionStore(text_issue=text_issue,prediction=prediction[0],probability=np.max(pred_proba[0]),date=datetime.datetime.now())
				
				with Session(engine) as session:
					session.add(pred_data)
					session.commit()

			with col2:
				plot_prediction_proba(text_issue)
				# st.help(st.metric)
			# st.metric(label='Accuracy',value='94',delta='0.2%')
			c1,c2,c3 = st.columns(3)
			c1.metric(label="Enhancement",value=probabilities['enhancement'],delta='{}%'.format(probabilities['enhancement']))
			c2.metric(label="Bug",value=probabilities['bug'],delta='{}%'.format(probabilities['bug']))
			c3.metric(label="Question",value=probabilities['question'],delta='{}%'.format(probabilities['question']))



	elif choice == "Monitor":
		st.subheader("Monitor")
		with Session(engine) as session:
			statement = select(PredictionStore)
			result = session.exec(statement).all()
			st.write(result)

	else:
		st.subheader("About")


if __name__ == '__main__':
	main()


