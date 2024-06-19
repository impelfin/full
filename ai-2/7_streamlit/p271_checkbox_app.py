import streamlit as st

st.title('스트림릿의 체크 박스 사용 예')

checked1 = st.checkbox('체크 박스 1')
st.write('체크 박스 1 Status: ', checked1)

if checked1:
    st.write('체크 박스 1 was checked')
else:
    st.write('체크 박스 1 was not checked')

checked2 = st.checkbox('체크 박스 2')
st.write('체크 박스 2 Status: ', checked2)

if checked2:
    st.write('체크 박스 2 was checked')
else:
    st.write('체크 박스 2 was not checked')

