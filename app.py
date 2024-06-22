import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2 as cv
import pandas as pd

def withtable(img):
    reader=easyocr.Reader(['en'],gpu=True)
    img1=img
    greyimg=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    ret,thresh=cv.threshold(greyimg,150,255,cv.THRESH_BINARY_INV)
    canny=cv.Canny(thresh,180,255)
    cont,hier=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    blank=np.zeros(img1.shape)

    large_cont=None
    large_Area=0
    large_box=None

    for cnt in cont:
        area=cv.contourArea(cnt)
        if area>large_Area:
            large_Area=area
            large_cont=cont
            large_box=cv.boundingRect(cnt)
    large_cont=cv.drawContours(img1.copy(),large_cont,-1,(0,255,0),1)
    x,y,w,h=large_box
    rect=cv.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
    st.image(rect,"↑ - table found image - ↑", clamp=True, channels='BGR')
    newimg=img1[y:y+h,x:x+w]
    rectimg=newimg.copy()
    st.image(rectimg,"↑ - cropped image - ↑", clamp=True, channels='BGR')
    try:
        result=reader.readtext(newimg)
        # print(result)
        for res in result:
            x1,y1=res[0][0]
            x2,y2=res[0][2]
            x1=int(x1)
            x2=int(x2)
            y1=int(y1)
            y2=int(y2)
            # print((x1,y1),"---",(x2,y2))
            text=res[1]
            conflevel=res[2]
            cv.rectangle(rectimg,(x1,y1),(x2,y2),(255,255,0),thickness=1)
            cv.putText(blank,text,(x1,y1+25),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(0,255,0),thickness=1)
        table = []
        cols = []
        col_ref = []
        x1 = []
        y1 = []
        for tup in range(len(result)):
            x, y = result[tup][0][0]
            x1.append(x)
            y1.append(y)
        for i in range(y1[0]-10, y1[0]+10):
            for j in range(len(x1)):
                if y1[j] == i:
                    cols.append(result[j][1])
                    col_ref.append(x1[j])
        # print(cols)
        for i in range(len(cols)):
            table.append([])
        # print(table)

        for tup in range(len(cols), len(result)): 
            for col_range in range(int(x1[tup])-30, int(x1[tup])+30):
                if x1[tup] == col_range:
                    for i in range(len(cols)):
                        if (col_range > (col_ref[i] - 100)) and (col_range < (col_ref[i] + 100)):
                            table[i].append(result[tup][1])

        df = pd.DataFrame(table).transpose()
        df = df.rename(columns = {i:cols[i] for i in range(len(cols))})

        st.image(rectimg,"↑ - text detected image - ↑", clamp=True)
        st.image(blank,"↑ - text extracted image - ↑", clamp=True)
        st.write("↓ - output DataFrame - ↓")
        st.write(df)
        st.toast("Successfully completed  Text Extraction Process")
        st.success('Successfully completed Text Extracytion Process', icon="✅")
    except Exception:
        st.error("It seems that the image you provided doesn't have any tables",icon="🚨")
        st.toast("Text Extraction Process failed")

def withouttable(img):
    try:
        reader=easyocr.Reader(['en'],gpu=True)
        img1=img
        imghei=img1.shape[0]
        newimghei=int(imghei-((25/100)*imghei))
        newimgheidown=int(imghei-((85/100)*imghei))
        finhei=imghei-newimghei
        finheidown=imghei-newimgheidown
        # print(imghei)
        # print(newimghei)
        croimg=img[finhei:finheidown,:]
        st.image(croimg,"↑ - cropped image - ↑",clamp=True)
        blank=np.zeros(croimg.shape)                                                                                                                                                                               
        reader=easyocr.Reader(['en'],gpu=True)
        result=reader.readtext(croimg)
        # print(result)
        for res in result:
            x1,y1=res[0][0]
            x2,y2=res[0][2]
            x1=int(x1)
            x2=int(x2)
            y1=int(y1)
            y2=int(y2)
            text=res[1]
            conflevel=res[2]
            # print((x1,y1),"---",(x2,y2))
            cv.rectangle(croimg,(x1,y1),(x2,y2),(255,255,0),thickness=1)
            cv.putText(blank,text,(x1,y1+10),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(0,255,0),thickness=1)
        st.image(croimg,"↑ - text detected image - ↑",clamp=True)
        st.image(blank,"↑ - text extracted image - ↑",clamp=True)
        table = []
        cols = []
        col_ref = []
        x1 = []
        y1 = []
        for tup in range(len(result)):
            x, y = result[tup][0][0]
            x1.append(x)
            y1.append(y)
        for i in range(y1[0]-10, y1[0]+10):
            for j in range(len(x1)):
                if y1[j] == i:
                    cols.append(result[j][1])
                    col_ref.append(x1[j])
        # print(cols)
        for i in range(len(cols)):
            table.append([])
        # print(table)

        for tup in range(len(cols), len(result)): 
            for col_range in range(int(x1[tup])-20, int(x1[tup])+20):
                if x1[tup] == col_range:
                    for i in range(len(cols)):
                        if (col_range > (col_ref[i] - 75)) and (col_range < (col_ref[i] + 75)):
                            table[i].append(result[tup][1])

        df = pd.DataFrame(table).transpose()

        df = df.rename(columns = {i:cols[i] for i in range(len(cols))})

        st.write("↓ - Output DataFrame - ↓")
        st.write(df)
        st.toast("Successfully completed  Text Extraction Process")
        st.success('Successfully completed Text Extraction Process', icon="✅")
    except Exception:
        st.error("It seems that the image you provided may have tables",icon="🚨")
        st.toast("Text Extraction Process failed")


def detectimage(inp):
    if(detbtn and type_det=='image with clear and visible table'):
        inparr=np.array(inp)
        withtable(inparr)

    elif(detbtn and type_det=="image without clear and visible table"):
        inparr=np.array(inp)
        withouttable(inparr)

st.set_page_config(page_title="OCR Text Extraction App")
st.title("Test Results Extraction from Medical Laboratory report images")
ctn=0
with st.sidebar:
    inp_file_sidebar=st.file_uploader("choose a report image",type=['png','jpg','jpeg','webp'],accept_multiple_files=False)
    "\n"
    "\n"
    if inp_file_sidebar is not None:
        st.image(inp_file_sidebar,"↑ - original image - ↑",clamp=True, channels='BGR')
        type_det=st.radio("choose image type",options=['image with clear and visible table',"image without clear and visible table"])
        detbtn=st.button("Analyze")
        if(inp_file_sidebar and detbtn):
            ctn=1
            
    else:
        st.subheader("please choose a image")

if(ctn==1):
    inpimg=Image.open(inp_file_sidebar)
    detectimage(inpimg)
