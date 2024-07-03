import streamlit as st
from PIL import Image
from model.model import query

# Set up the Streamlit app
st.title('Text to Image Streamlit App')

# Text input
user_input = st.text_input('Enter some text:')

# Submit button
if st.button('Submit'):
    # Display the input text
    st.write(f'You entered: {user_input}')

    images = query(user_input)
    
    # Load and display the image
    image = Image.open(images[0][0])
    st.image(image, caption=images[0][1], use_column_width=True)

