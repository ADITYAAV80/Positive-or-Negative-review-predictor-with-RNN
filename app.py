import pickle
import streamlit as st
import numpy as np
import torch
from datasets import load_dataset
from rnn import RNN
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore



if __name__=="__main__":

    with open("tokenizer.pickle","rb") as handler:
        tokenizer = pickle.load(handler)

    st.title(":orange[Movie positive or negative review predictor]")

    st.subheader(":orange[Vanilla RNN for Text Classification]")
    st.write(
        """
        :orange[
        This application demonstrates a Recurrent Neural Network (RNN) designed for binary text classification. The model processes sequential text data using an embedding layer followed by a simple RNN layer, making predictions based on learned patterns.
        ]
        """)
    
    st.subheader(":orange[Model Overview]")
    st.write(
        """
        :rainbow[Embedding Layer: Converts words into dense vector representations.]\n
        :rainbow[RNN Layer: Captures sequential dependencies in text data.]\n
        :rainbow[Linear Layer: Maps the final RNN output to a single prediction score.]\n
        :rainbow[Activation: Uses sigmoid to generate probability scores.]\n
        :rainbow[Loss Function: Binary Cross Entropy with Logits (BCEWithLogitsLoss).]\n
        :rainbow[Optimizer: Adam with a learning rate of 0.005.]\n
        """)
    
    st.subheader(":orange[Training Summary]")
    
    st.write(
        """
        :orange[The model was trained for 8 epochs, improving accuracy from 52.6% to 83.6% on the training set and achieving up to 71.9% accuracy on the test set. Loss values fluctuated slightly, indicating room for further optimization.]\n\n
        :orange[📊 Try entering text to see the model's classification in action! 🚀]\n
        """
    )

    review = st.text_area(label="Enter your review")

    # Set the background image using custom CSS
    background_url = "https://images.unsplash.com/photo-1616530940355-351fabd9524b?q=80&w=1035&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        
        background-blend-mode: overlay;
        background-color: rgba(0, 0, 0, 0.7);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)


    X_input_seq = tokenizer.texts_to_sequences([review])
    X_input_padded = pad_sequences(X_input_seq, maxlen=300, padding="pre")
    X_input_tensor = torch.tensor(X_input_padded)

    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index

    rnn = RNN(vocab_size)
    rnn.load_state_dict(torch.load("rnn.pth",weights_only=True))

    if (review is None) or (review==""):
        st.write("Review not populated yet")
    else:
        rnn.eval()
        with torch.inference_mode():
            y_input_logits = rnn(X_input_tensor)

            y_test_pred = torch.sigmoid(y_input_logits)

            if float(y_test_pred) > 0.5:
                st.write(f":green[Positive Review with probability {float(y_test_pred):.2f}]")
            else:
                st.write(f":red[Negative review with probability {1-float(y_test_pred):.2f}]")
        