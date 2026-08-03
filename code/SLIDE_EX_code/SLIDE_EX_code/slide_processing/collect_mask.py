import os,sys
import numpy as np
import pandas as pd
from PyPDF2 import PdfMerger, PdfReader


project = sys.argv[1]
path2project = f"./{project}/"  # TODO: substitute with your own filepath
path2meta = path2project + "metadata.csv"  # TODO: substitute with your own filepath
path2inputs = path2project + "mask/"
print(f"path2inputs: {path2inputs}")

# find num files within a folder
slide_names = []
for f in os.listdir(path2inputs):
    if f.endswith(".pdf"):
        slide_names.append(f)
len(slide_names)

# alphabet sort
slide_names = sorted(slide_names)
slide_names[0]

# merge
mergedObject = PdfMerger()
for slide_name in slide_names:    
    mergedObject.append(PdfReader("%s%s"%(path2inputs, slide_name), "rb"))

mergedObject.write("%spbcp_mask.pdf"%path2project)

print("--- completed collecting mask--- ")
