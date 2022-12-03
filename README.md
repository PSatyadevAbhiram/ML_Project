# ML_Project

# You wil need to download the pkl from 
# https://github.com/kennymckormick/pyskl/blob/main/tools/data/data_doc.md?fbclid=IwAR2zZlhPAbGdXd9859fA2FV_razbGp67FX62GZdn6mGVtqYBccecVHaUkLA

Once you visit the page from the link above, please download NTURGB+D 120 .pkl file and place it in your cwd. Run gen_ml_features.py file 
to generate all the features. This might take a while.

Next step: Use multiprocessing as described in
https://docs.python.org/3/library/multiprocessing.html?fbclid=IwAR0x1hdAiIxpV95kMC6DA5F-4T7rFDxlHi6w-JwwJW-BHgqqQoEE5Ha7hiI
to make things faster. 

After that: We must train the retrieved features on an ML model and retrive the accuracy. From this, we can decide where to go next.
