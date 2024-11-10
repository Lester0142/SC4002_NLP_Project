# SC4002_NLP_Project
Deadline: Sunday, 10th November\
Deliverable:
1. Report (SC4002_G12.pdf)
   - cover page: all group members name & contributions
   - design & answer to the questions
2. README.txt
   - instructions to run code
   - explanation of sample output from code
3. Zip File
   - Python source codes

# Instructions
1. Ensure that system has python installed
```
python --version
```
2. Create a virtual environment and download relevant packages \

windows:
```
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt
```
unix: 
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the Basic UI to view accuracy result from the model
```
cd src
python main.py
```
* takes awhile to run, below is what the UI will look like when it is ready

Example:

Console:\
Select a model type to test from the following options ('q' to exit):\
1: RNN_BASE\
2: RNN\
3: RNN_OOV\
4: LSTM\
5: GRU\
6: CNN\
Enter the number corresponding to your choice:

Input: 
1

Ouput:\
Testing model: RNN_BASE\
testing accuracy: 0.7533

Notes
------
1. For Part 3.5, as it is significantly different from the other parts, we have not added it to this script. The results can be seen by running sentiment_bert.ipynb in folder part_3 directly.
2. For all the other models, it is also possible to see the training and preparation codes as well as run them directly by finding the appropriate file (e.g. sentimentRNN_updated_embeddings_with_OOV_handling.ipynb in folder part_3 for RNN_OOV). When doing so, take caution not to run the training cell unintentionally as this will take sometime.
3. You may encounter an error saying the word2vector model file is missing. To rectify this, run load_model.py under part_1.



