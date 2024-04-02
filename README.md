 create venv:
 python -m venv env
 env/Scripts/activate
 pip install -r requirements.txt

 run project:
    For plot showing original flight and filtered flight + measure error of the filtered positions: 
     - python project2/project2.py
    For heatmaps and calculations (task 8 and 9), to use smoothing edit lines 35 and 36:
    - python project2/project2_calculations_heatmaps.py
