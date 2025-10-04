**FaceVerificationAppPoliticians**

---

A face verification system for politicians using FastAPI, Streamlit, and MongoDB.


***Setup***



Install Python 3.8+ and MongoDB Community Edition.

Install MongoDB Database Tools for mongoexport/mongoimport.


***Create and activate a virtual environment:***
python -m venv venv

venv\\Scripts\\activate





***Install dependencies:***
pip install -r requirements.txt

pip install streamlit





Set up .env based on .env.example.


***Start MongoDB:***
mongod





***Populate the database:***
python update\_embeddings.py



Requires a dataset folder with politician subfolders containing .jpg images.


***(Optional) Import sample data:***
mongoimport --db face\_verification\_db --collection politicians --file sample\_data/politicians.json





***Run backend:***
python api.py





***Run frontend:***
streamlit run frontend.py







**Files**



*api.py*: FastAPI backend with MongoDB connection management.

*frontend.py*: Streamlit frontend for GUI interaction.

*config.py*: Configuration settings.

*requirements.txt*: Backend dependencies.

*update\_embeddings.py*: Populates MongoDB with face embeddings.

*verify\_faces.py*: Standalone webcam verification script.

*test\_camera\_stability.py*: Camera testing script.

*.env.example*: Template for environment variables.

*sample\_data/politicians.json*: Sample MongoDB data.



**Notes**



MongoDB connections are opened on demand and closed when not in use.

The dataset and test folders are excluded from Git due to potential sensitivity/size.

Ensure a webcam is available for real-time verification.



