import streamlit as st
import pickle
from PIL import Image


def main():
    st.title('Lung cancer prediction')
    img=Image.open('lung_img.png')
    st.image(img,width=600)
    age=st.number_input('enter age:')
    smoke=st.number_input('enter smoke:')
    area=st.number_input('enter AreaQ:')
    alcohol=st.number_input('enter Alcohol:')
    feature=[age,smoke,area,alcohol]
    knn=pickle.load(open('knn_model1.sav','rb'))
    s=pickle.load(open('scaler_knn1.sav','rb'))
    pred=st.button('predict')
    if pred:
        result=knn.predict(s.transform([feature]))
        result=int(result[0])
        if result == 0:
            st.write('not cancer patient')
        else:
            st.write('cancer patient')


main()   #fun calling
