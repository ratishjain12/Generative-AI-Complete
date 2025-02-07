import streamlit as st
from colivara_py import ColiVara
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

load_dotenv()

model = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.5, max_tokens = 1024)

client = ColiVara(
    api_key= os.getenv("COLIVARA_API_KEY"),
    base_url="https://api.colivara.com"
)

def display_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    st.image(img)


def main():
    load_dotenv()
    st.set_page_config(page_title="Colivara Multimodal RAG")
    st.title("Colivara Vision Based Multimodal RAG Application")
    query = st.text_input("Enter your query here:")
    if query:
        response = client.search(query=query,top_k=1,collection_name="embedding_collection")
        display_base64_image(response.results[0].img_base64)

        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + response.results[0].img_base64}},
            ],
        )

        response  = model.invoke([message])
        st.write(response.content)

    with st.sidebar:
        st.subheader("upload file")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if st.button("Process"):
            with st.spinner("processing..."):
                if uploaded_file is not None:
                    encoded_content = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                    client.upsert_document(
                        name=uploaded_file.name,
                        document_base64=encoded_content,
                        collection_name="embedding_collection",
                        wait=True
                    )
                    st.write(f"upserted: {uploaded_file.name}")
                else:
                    st.toast("Please upload a PDF file")
        



if __name__ == "__main__":
    main()