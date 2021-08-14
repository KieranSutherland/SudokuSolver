## Demo
Here's a real time demo of the program viewing three different puzzles, tracking the grid, solving them and placing the solved numbers where they should be relative to the grid:
![demo](sudoku-ar-demo.gif)
## Requirements
Make sure you have Python installed locally, which can be downloaded [here](https://www.python.org/downloads/).
## Running locally:
The requirements.txt file is for Heroku deployment **only**. Use requirements_local.txt for local usage:
```pip install -r requirements_local.txt```
If you want to retrain the tensorflow model:
```python digits_model/trainer.py```
If you want to run the application directly:
```python main.py```
If you want to run the application through a Flask server:
```python -m flask run```
## Know issues
You may also find that the application quits out almost immediately, if it does, try commenting out the yield returns in main.py (I'll find a proper fix for this at some point)