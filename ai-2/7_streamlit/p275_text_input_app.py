import streamlit as st

st.title("스트림릿의 텍스트 입력 사용 예")

user_id = st.text_input('아이디(ID) 입력', value='streamlit', max_chars=15)
user_password = st.text_input('패스워드(Password) 입력', value='abcd', type='password')

if user_id == "streamlit":
    if user_password == "1234":
        st.write('로그인 성공')
    else:
        st.write('패스워드가 틀렸습니다. 다시 입력하세요.')
else:
    st.write('없는 ID입니다. 회원가입을 하거나 올바른 ID를 다시 입력하세요.')

