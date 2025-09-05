Guide to download dataset:
1. Go to https://zenodo.org/records/3713280 and download  pan12-sexual-predator-identification-test-and-training.zip
2. Inside the zip are two zips : one for training and one for testing
### Required Files
- `pan12-sexual-predator-identification-training-corpus-2012-05-01.xml`  
- `pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt`  
- `pan12-sexual-predator-identification-test-corpus-2012-05-17.xml`  

The `model_training.py` script is used to train and save the model.  
Once run, it will generate a `model_save/` directory. Use this along with `pyserver.py` to run the server.

After starting the server, you can test predictions from the terminal with:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text": "insert example text"}'


