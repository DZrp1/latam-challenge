import unittest

from fastapi.testclient import TestClient
from challenge import application as app


class TestBatchPipeline(unittest.TestCase):
    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        with TestClient(app) as client:
            response = client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        with TestClient(app) as client:
            response = client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        with TestClient(app) as client:
            response = client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        with TestClient(app) as client:
            response = client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)