import streamlit as st
import os
import numpy as np
from PIL import Image
from pathlib import Path
import seam_carving
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Resize Image through Seam Carving")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    directory = "tempDir"
    path = os.path.join(os.getcwd(), directory)
    p = Path(path)
    if not p.exists():
        os.mkdir(p)
    with open(os.path.join(path, uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer()) 
    file_loc = os.path.join(path, uploaded_file.name)  
    origin = Image.open(uploaded_file)
    origin.save('demo.png',quality = 50,optimize=True)
    src = np.array(origin)
    src_h, src_w, _ = src.shape
    cap="Uploaded Image. Height="+str(src_h)+ " Width=" + str(src_w)
    st.image(image, caption=cap, use_column_width=False)
    set_h = st.text_input("Enter new Height ")
    set_w = st.text_input("Enter new Width ")
    if st.button("Resize"):
        with st.spinner('''Processing Image! 
        Applying Algorithm............'''):
            if set_h and set_w:
                set_h=int(set_h)
                set_w=int(set_w)
                dst = seam_carving.resize(
                    src, (set_w,set_h),
                    energy_mode='forward',   # Choose from {backward, forward}
                    order='height-first',  # Choose from {width-first, height-first}
                    keep_mask=None
                )
                dst_h,dst_w, _ =dst.shape 
                im = Image.fromarray(dst)
                result_cap = "Resized Image. Height="+str(dst_h) +" Width=" +str(dst_w)
                st.image(im, caption=result_cap, use_column_width=False)
        st.success('Done! Image is ready')
