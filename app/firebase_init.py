import firebase_admin
from firebase_admin import credentials, firestore, storage, auth

cred = credentials.Certificate('/Users/mohamedabdelwahab/Documents/datapal-fa463-firebase-adminsdk-u466b-530cb89324.json')
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'datapal-fa463.appspot.com'
    })

db = firestore.client()
bucket = storage.bucket()